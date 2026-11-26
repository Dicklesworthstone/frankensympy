"""Scalar Newton uses bounded damping without weakening source or failure checks."""

import importlib
import math

import pytest
import sympy as s


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_log_backtracks_into_its_original_real_domain(budgeted):
    x = s.Dummy("scalar_log_coordinate")
    budget = s.EvaluationBudget() if budgeted else None
    # The ordinary full step is outside log's real domain.
    assert 10.0 - math.log(10.0) / 0.1 < 0.0
    root = s.nsolve(s.log(x), x, 10, budget=budget)
    assert type(root) is s.Float
    assert float(root) == pytest.approx(1.0, abs=1e-12)
    assert abs(math.log(float(root))) <= 1e-12


@pytest.mark.parametrize("initial", [-1, 1])
@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_damping_breaks_an_exact_two_cycle(initial, budgeted):
    x = s.Symbol("scalar_cycle_coordinate")
    source = x**3 - 5*x
    # Undamped Newton alternates exactly between +1 and -1. Half of the
    # rejected step instead reaches the actual root zero in one iteration.
    first = initial - (initial**3 - 5*initial) / (3*initial**2 - 5)
    second = first - (first**3 - 5*first) / (3*first**2 - 5)
    assert first == -initial and second == initial
    budget = s.EvaluationBudget() if budgeted else None
    result, = s.nsolve(source, x, initial, maxiter=1, dict=True, budget=budget)
    assert list(result) == [x]
    assert float(result[x]) == 0.0


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_damping_rejects_overflowing_exponential_trials(budgeted):
    x = s.Symbol("scalar_exp_coordinate")
    budget = s.EvaluationBudget() if budgeted else None
    full_trial = -10.0 - (math.exp(-10.0) - 1) / math.exp(-10.0)
    with pytest.raises(OverflowError):
        math.exp(full_trial)
    root = float(s.nsolve(s.exp(x) - 1, x, -10, budget=budget))
    assert root == pytest.approx(0.0, abs=1e-12)
    assert abs(math.expm1(root)) <= 1e-12


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_damping_composes_with_checked_inverse_derivatives(budgeted):
    x = s.Symbol("scalar_atan_coordinate")
    budget = s.EvaluationBudget() if budgeted else None
    root = float(s.nsolve(s.atan(x), x, 10, budget=budget))
    assert root == pytest.approx(0.0, abs=1e-12)
    assert abs(math.atan(root)) <= 1e-12


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_initial_source_poles_survive_held_zero_factors(budgeted):
    x = s.Symbol("scalar_original_pole")
    source = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    budget = s.EvaluationBudget() if budgeted else None
    with pytest.raises(ValueError):
        s.nsolve(source, x, 0, budget=budget)


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_backtracking_does_not_accept_a_nonroot_stationary_residual(budgeted):
    x = s.Symbol("scalar_no_real_root")
    budget = s.EvaluationBudget() if budgeted else None
    with pytest.raises(ValueError):
        s.nsolve(1 + (10**30*x)**2, x, 1e-30, budget=budget)


def test_scalar_retry_requires_closed_sources_without_custom_callbacks():
    x = s.Symbol("scalar_custom_coordinate")
    calls = []

    class UserLaw(s.Function):
        def _eval_evalf(self, precision):
            calls.append(("evalf", precision))
            raise AssertionError("an unregistered source callback must not run")

        def _eval_derivative(self, variable):
            calls.append(("derivative", variable))
            raise AssertionError("an unregistered derivative callback must not run")

    source = UserLaw(x, evaluate=False)
    with pytest.raises(NotImplementedError):
        s.nsolve(source, x, 10)
    assert calls == []


@pytest.mark.parametrize("failure", [
    ValueError("numeric power exceeds the integer result limit"),
    ValueError("unexpected closed-program invariant failure"),
    RuntimeError("math domain error"),
    MemoryError("numeric allocation failure"),
])
def test_scalar_math_trial_limits_and_unknown_failures_are_terminal(monkeypatch, failure):
    numerical_math = importlib.import_module("sympy.utilities.lambdify")
    original = numerical_math._evaluate_math
    negative_trials = []

    def evaluate(program, inputs):
        if inputs[0] < 0.0:
            negative_trials.append(inputs[0])
            raise failure
        return original(program, inputs)

    monkeypatch.setattr(numerical_math, "_evaluate_math", evaluate)
    x = s.Symbol("scalar_math_failure")
    with pytest.raises(type(failure)) as caught:
        s.nsolve(s.log(x), x, 10)
    assert caught.value is failure
    assert len(negative_trials) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_scalar_domain_retry_preserves_typed_operational_failures(monkeypatch, cancel):
    numerical = importlib.import_module("sympy.solvers.numerical")
    original = numerical._scalar_newton_functions
    negative_trials = []

    def functions(source, variable, budget):
        residual, derivative = original(source, variable, budget)

        def guarded(value, *, budget):
            if value < 0.0:
                negative_trials.append(value)
                if cancel:
                    budget.cancel()
                else:
                    budget._charge(2**64 - 1)
            return residual(value, budget=budget)

        return guarded, derivative

    monkeypatch.setattr(numerical, "_scalar_newton_functions", functions)
    x = s.Symbol("scalar_operational_failure")
    budget = s.EvaluationBudget()
    expected = s.EvaluationCancelled if cancel else s.EvaluationBudgetExceeded
    with pytest.raises(expected):
        s.nsolve(s.log(x), x, 10, budget=budget)
    assert len(negative_trials) == 1


def test_scalar_damped_solve_spends_one_cumulative_budget():
    x = s.Symbol("scalar_shared_budget")
    measured = s.EvaluationBudget()
    root = s.nsolve(s.log(x), x, 10, budget=measured)
    cost = 10_000_000 - measured.remaining_steps
    assert float(root) == pytest.approx(1.0, abs=1e-12)
    assert cost > 0

    exact = s.EvaluationBudget(max_steps=cost)
    assert float(s.nsolve(s.log(x), x, 10, budget=exact)) == pytest.approx(1.0, abs=1e-12)
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(s.log(x), x, 10, budget=exact)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(s.log(x), x, 10, budget=s.EvaluationBudget(max_steps=cost - 1))


def test_scalar_trial_and_evaluation_caps_remain_independent(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_bounded_cycle")
    source = x**3 - 5*x

    with monkeypatch.context() as patch:
        patch.setattr(numerical, "_MAX_LINE_SEARCH_STEPS", 1)
        with pytest.raises(ValueError, match="line search"):
            s.nsolve(source, x, 1)

    # Initial residual, derivative and rejected full step consume all three
    # evaluations. The successful half-step cannot bypass that exhausted cap.
    with monkeypatch.context() as patch:
        patch.setattr(numerical, "_MAX_SCALAR_EVALUATIONS", 3)
        with pytest.raises(ValueError, match="numerical evaluation budget"):
            s.nsolve(source, x, 1)
