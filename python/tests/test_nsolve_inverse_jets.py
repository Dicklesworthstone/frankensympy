"""Scalar Newton retains finite inverse-function slopes at extreme inputs."""

import importlib
import math

import pytest
import sympy as s


def forbid_symbolic_derivatives(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("an admitted inverse-function source must use checked numerical jets")

    monkeypatch.setattr(numerical, "diff", forbidden)
    return numerical


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("sign", (-1, 1))
def test_scalar_asinh_solves_finite_extreme_roots_without_squaring_the_argument(
    budgeted, sign, monkeypatch,
):
    forbid_symbolic_derivatives(monkeypatch)
    x = s.Symbol("scalar_asinh_extreme")
    target = sign*461
    budget = s.EvaluationBudget() if budgeted else None
    root = s.nsolve(s.asinh(x)-target, x, sign*1e200, budget=budget)
    assert type(root) is s.Float
    value = float(root)
    assert math.isfinite(value)
    assert value == pytest.approx(math.sinh(target), rel=3e-13)
    assert abs(math.asinh(value)-target) <= 1e-12


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("initial", (-1e200, 1e200))
def test_scalar_atan_composition_recovers_from_a_finite_extreme_initial_value(
    budgeted, initial, monkeypatch,
):
    forbid_symbolic_derivatives(monkeypatch)
    x = s.Symbol("scalar_atan_extreme")
    budget = s.EvaluationBudget() if budgeted else None
    root = float(s.nsolve(s.atan(x)+x-1, x, initial, budget=budget))
    assert 0 < root < 1
    assert abs(math.atan(root)+root-1) <= 1e-12


@pytest.mark.parametrize("function", (s.atan, s.asinh))
@pytest.mark.parametrize("budgeted", (False, True))
def test_inverse_function_initial_zero_roots_do_not_evaluate_a_jacobian(
    function, budgeted, monkeypatch,
):
    numerical = forbid_symbolic_derivatives(monkeypatch)
    original_compile = numerical.compile_residual_system

    class NoDerivative:
        def jacobian(self, *args, **kwargs):
            raise AssertionError("an exact-zero residual must return before derivative evaluation")

    def compile_without_evaluation(*args, **kwargs):
        original_compile(*args, **kwargs)
        return NoDerivative()

    monkeypatch.setattr(numerical, "compile_residual_system", compile_without_evaluation)
    x = s.Symbol("scalar_inverse_zero")
    budget = s.EvaluationBudget() if budgeted else None
    assert float(s.nsolve(function(x), x, 0, budget=budget)) == 0


@pytest.mark.parametrize("function", (s.atan, s.asinh))
@pytest.mark.parametrize("exception", (ValueError, RuntimeError))
def test_inverse_function_checked_compile_refusals_never_select_symbolic_fallback(
    function, exception, monkeypatch,
):
    numerical = forbid_symbolic_derivatives(monkeypatch)
    refusal = exception("checked inverse-function compilation refused")

    def refused(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(numerical, "compile_residual_system", refused)
    x = s.Symbol("scalar_inverse_refusal")
    with pytest.raises(exception) as caught:
        s.nsolve(function(x)+x-1, x, 0)
    assert caught.value is refusal


@pytest.mark.parametrize("function", (s.atan, s.asinh))
@pytest.mark.parametrize("budgeted", (False, True))
def test_inverse_function_jets_cannot_erase_held_original_source_poles(
    function, budgeted, monkeypatch,
):
    forbid_symbolic_derivatives(monkeypatch)
    x = s.Symbol("scalar_inverse_original_pole")
    reciprocal = s.Pow(x, -1, evaluate=False)
    source = s.Add(x, s.Mul(0, function(reciprocal, evaluate=False), evaluate=False),
                   evaluate=False)
    budget = s.EvaluationBudget() if budgeted else None
    with pytest.raises(ValueError):
        s.nsolve(source, x, 0, budget=budget)


@pytest.mark.parametrize("function", (s.atan, s.asinh))
def test_inverse_function_dispatch_preserves_large_exact_integer_intermediates(
    function, monkeypatch,
):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe integer intermediates must stay on the admitted math route")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    x = s.Symbol("scalar_inverse_integer_source")
    huge = s.Mul(*(s.Integer(1 << 40) for _ in range(32)), evaluate=False)
    source = s.Add(huge, s.Mul(-1, huge, evaluate=False), function(x), x, -1,
                   evaluate=False)
    root = float(s.nsolve(source, x, 0))
    native_function = math.atan if function is s.atan else math.asinh
    assert abs(native_function(root)+root-1) <= 1e-12


@pytest.mark.parametrize("function", (s.atan, s.asinh))
def test_inverse_function_scalar_work_keeps_resource_and_cancellation_failures(
    function, monkeypatch,
):
    forbid_symbolic_derivatives(monkeypatch)
    x = s.Symbol("scalar_inverse_budget")
    source = function(x)+x-1
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(source, x, 0, budget=s.EvaluationBudget(max_steps=4))
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        s.nsolve(source, x, 0, budget=cancelled)
