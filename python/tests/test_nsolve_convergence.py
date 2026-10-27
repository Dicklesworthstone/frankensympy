"""Scalar numerical convergence must check the equation, not just a step."""

import math

import pytest

from sympy import Expr, Float, Symbol, exp, nsolve, sqrt


def test_nsolve_is_exported_through_the_standard_solver_modules():
    from sympy.solvers import nsolve as package_nsolve
    from sympy.solvers.solvers import nsolve as module_nsolve

    assert package_nsolve is nsolve
    assert module_nsolve is nsolve


def test_scalar_newton_converges_with_a_small_residual():
    x = Symbol("not a Python identifier")
    root = nsolve(x**2 - 2, x, 1)
    assert type(root) is Float
    assert float(root) == pytest.approx(math.sqrt(2), abs=1e-12)
    assert abs(float(root)**2 - 2) <= 1e-12
    assert float(nsolve(x - 2, x, 0, maxiter=1)) == 2.0


def test_initial_multiple_and_singular_derivative_roots_are_accepted():
    x = Symbol("x")
    assert float(nsolve(x**2, x, 0)) == 0.0
    assert float(nsolve(sqrt(x), x, 0)) == 0.0


def test_tiny_nonzero_derivatives_do_not_reject_scaled_equations():
    x = Symbol("x")
    assert float(nsolve((x - 2) / 10**50, x, 0)) == 2.0


def test_tiny_step_with_large_residual_does_not_converge():
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve(1 + (10**30*x)**2, x, 1e-30)


def test_binary64_stagnation_with_nonzero_residual_refuses():
    x = Symbol("x")
    with pytest.raises(ValueError, match="stagnat"):
        nsolve(exp(10**20*(x - 1)) + 1, x, 1)


def test_nonfinite_residual_and_newton_step_refuse():
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve(exp(x), x, 1000)
    with pytest.raises(ValueError, match="Newton step"):
        nsolve(x / 10**308 + 2, x, 1e308)
    with pytest.raises(ValueError):
        nsolve(sqrt(x), x, -1)


@pytest.mark.parametrize("initial", [math.inf, -math.inf, math.nan])
def test_initial_value_must_be_finite(initial):
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve(x - 1, x, initial)


@pytest.mark.parametrize("tolerance", [0, -1, math.inf, math.nan])
def test_tolerance_must_be_positive_and_finite(tolerance):
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve(x - 1, x, 0, tol=tolerance)


@pytest.mark.parametrize("iterations", [0, -1, 10_001])
def test_iteration_budget_is_bounded(iterations):
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve(x - 1, x, 0, maxiter=iterations)


def test_iteration_count_and_numeric_inputs_do_not_invoke_custom_hooks():
    x = Symbol("x")

    class CustomFloat:
        def __float__(self):
            raise AssertionError("unregistered numeric conversion hook")

    for value in ("1", CustomFloat(), 1j):
        with pytest.raises(TypeError):
            nsolve(x - 1, x, value)
    with pytest.raises(TypeError):
        nsolve(x - 1, x, 0, maxiter=1.5)
    with pytest.raises(ValueError, match="failed to converge"):
        nsolve(x**2 - 2, x, 1, maxiter=1)


def test_newton_compilation_and_iteration_do_not_print_expressions(monkeypatch):
    x = Symbol("x")
    expression = x**2 - 2

    def forbidden(*args, **kwargs):
        raise AssertionError("printing cannot bind numerical variables")

    monkeypatch.setattr(Expr, "__str__", forbidden)
    monkeypatch.setattr(Symbol, "__str__", forbidden)
    assert float(nsolve(expression, x, 1)) == pytest.approx(math.sqrt(2), abs=1e-12)
