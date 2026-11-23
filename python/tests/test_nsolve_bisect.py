"""Bracketed numerical candidates retain sources, residual checks and budgets."""

import importlib
import math
import sys

import pytest
import sympy as s


LIMIT = 1 << 40


def ledger(**changes):
    limits = dict(max_steps=LIMIT, max_memory_bytes=LIMIT, max_allocations=LIMIT)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


@pytest.mark.parametrize("native", [False, True])
def test_polynomial_and_transcendental_brackets_meet_both_tolerances(native):
    x = s.Symbol("bracketed coordinate, not source code")
    budget = ledger() if native else None
    root = s.nsolve(x*x-2, x, (0, 2), solver="bisect", budget=budget)
    assert type(root) is s.Float
    assert float(root) == pytest.approx(math.sqrt(2), rel=0, abs=1e-12)
    assert abs(float(root)**2-2) <= 1e-12
    root = float(s.nsolve(s.cos(x)-x, x, [0, 1], solver="bisect", budget=budget))
    assert root == pytest.approx(0.7390851332151607, rel=0, abs=1e-12)
    assert abs(math.cos(root)-root) <= 1e-12


def test_bracket_recovers_from_a_zero_derivative_and_a_newton_cycle():
    x = s.Symbol("bracketed_failure_recovery")
    with pytest.raises(ValueError, match="derivative is zero"):
        s.nsolve(x**3-2, x, 0)
    root = float(s.nsolve(x**3-2, x, (0, 2), solver="bisect"))
    assert abs(root**3-2) <= 1e-12
    cycling = x**3-2*x+2
    with pytest.raises(ValueError, match="failed to converge"):
        s.nsolve(cycling, x, 0, maxiter=8)
    root = float(s.nsolve(cycling, x, (-2, 0), solver="bisect"))
    assert -2 <= root <= 0
    assert abs(root**3-2*root+2) <= 1e-12


@pytest.mark.parametrize("native", [False, True])
def test_endpoint_and_multiple_roots_do_not_compile_or_evaluate_derivatives(native, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("bracketed_endpoint")

    def forbidden(*args, **kwargs):
        raise AssertionError("bisection must not request derivatives or a different solver")

    monkeypatch.setattr(numerical, "diff", forbidden)
    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    monkeypatch.setattr(numerical, "_scalar_newton_functions", forbidden)
    budget = ledger() if native else None
    assert float(s.nsolve(s.sqrt(x), x, (0, 4), solver="bisect", budget=budget)) == 0
    assert float(s.nsolve(x-2, x, (0, 2), solver="bisect", budget=budget)) == 2
    assert float(s.nsolve(x**3, x, (-1, 1), solver="bisect", budget=budget)) == 0


@pytest.mark.parametrize("native", [False, True])
def test_zero_left_endpoint_does_not_hide_an_invalid_right_endpoint(native):
    x = s.Symbol("bracketed_zero_endpoint_domain")
    source = s.Mul(x, s.sqrt(1-x), evaluate=False)
    published = []
    with pytest.raises(ValueError):
        published.append(s.nsolve(source, x, (0, 2), solver="bisect",
                                  budget=ledger() if native else None))
    assert published == []


def test_scalar_equality_mapping_preserves_the_original_dummy():
    x = s.Dummy("bracketed_same_name")
    other = s.Dummy("bracketed_same_name")
    result = s.nsolve(s.Eq(x*x, 2), x, (0, 2), solver="bisect", dict=True)
    assert type(result) is list and len(result) == 1
    assert len(result[0]) == 1 and next(iter(result[0])) is x
    assert other not in result[0]
    assert type(result[0][x]) is s.Float
    assert abs(float(result[0][x])**2-2) <= 1e-12


def test_same_printed_names_do_not_bind_a_foreign_symbol():
    x, other = s.Symbol("bracketed_x"), s.Symbol("bracketed_x", positive=True)
    with pytest.raises(ValueError, match="unbound"):
        s.nsolve(other-1, x, (0, 2), solver="bisect")


@pytest.mark.parametrize("native", [False, True])
def test_sign_comparison_and_position_accuracy_survive_underflowing_products(native):
    x = s.Symbol("bracketed_tiny_residual")
    budget = ledger() if native else None
    # Endpoint products underflow to signed zero. Residuals are already below
    # tol at both endpoints, but the reported coordinate must still converge.
    source = (x**3-2)/10**200
    root = float(s.nsolve(source, x, (0, 2), solver="bisect", budget=budget))
    assert root == pytest.approx(2**(1/3), rel=0, abs=1e-12)
    with pytest.raises(ValueError, match="opposite signs"):
        s.nsolve((x*x+1)/10**200, x, (-1, 1), solver="bisect", budget=budget)


def test_midpoint_does_not_overflow_for_finite_extreme_bounds():
    x = s.Symbol("bracketed_extreme")
    assert float(s.nsolve(x, x, (-sys.float_info.max, sys.float_info.max),
                         solver="bisect")) == 0
    assert float(s.nsolve(x-s.Float(1.3e308), x, (1e308, 1.6e308),
                         solver="bisect")) == 1.3e308


def test_large_endpoint_residuals_do_not_require_a_finite_product():
    x = s.Symbol("bracketed_large_residual")
    root = s.nsolve(10**200*(x-s.Rational(3, 8)), x, (0, 1), solver="bisect")
    assert float(root) == 3/8


@pytest.mark.parametrize("native", [False, True])
def test_poles_are_not_roots_even_after_the_interval_becomes_tiny(native):
    x = s.Symbol("bracketed_pole")
    for source in (1/x, 1/(x-s.Rational(1, 3))):
        published = []
        with pytest.raises(ValueError):
            published.append(s.nsolve(source, x, (-1, 1), solver="bisect",
                                      budget=ledger() if native else None))
        assert published == []


@pytest.mark.parametrize("native", [False, True])
def test_a_jump_without_a_zero_does_not_pass_a_width_only_stopping_rule(native):
    x = s.Symbol("bracketed_discontinuity")
    source = s.Piecewise((-1, x < 0), (1, True))
    with pytest.raises(ValueError, match="failed to converge"):
        s.nsolve(source, x, (-1, 1), solver="bisect", maxiter=100,
                 budget=ledger() if native else None)


@pytest.mark.parametrize("native", [False, True])
def test_held_original_poles_survive_zero_products_and_equations(native):
    x = s.Symbol("bracketed_original_domain")
    hidden = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.Add(x, hidden, evaluate=False)
    for expression in (source, s.Eq(source, 0, evaluate=False)):
        with pytest.raises(ValueError):
            s.nsolve(expression, x, (-1, 1), solver="bisect",
                     budget=ledger() if native else None)


def test_real_domain_failure_is_terminal_and_endpoint_sources_are_checked():
    x = s.Symbol("bracketed_domains")
    root = float(s.nsolve(s.log(x)-1, x, (1, 4), solver="bisect"))
    assert root == pytest.approx(math.e, rel=0, abs=1e-12)
    for source, bounds in ((s.log(x)-1, (-1, 4)), (s.sqrt(x), (-1, 4)),
                           (s.exp(x), (700, 710)), (1/x, (-1, 0))):
        with pytest.raises(ValueError):
            s.nsolve(source, x, bounds, solver="bisect")


def test_bisection_only_evaluates_inside_the_captured_bracket(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    original = numerical.lambdify
    evaluated = []

    def observe(*args, **kwargs):
        function = original(*args, **kwargs)

        def call(value):
            evaluated.append(value)
            return function(value)

        return call

    monkeypatch.setattr(numerical, "lambdify", observe)
    x = s.Symbol("bracketed_invariant")
    root = float(s.nsolve(s.log(x)-1, x, (1, 4), solver="bisect"))
    assert abs(math.log(root)-1) <= 1e-12
    assert evaluated[:2] == [1, 4]
    assert len(evaluated) > 2 and all(1 <= value <= 4 for value in evaluated)


@pytest.mark.parametrize("budgeted", [False, True])
def test_iteration_exhaustion_and_binary64_stagnation_do_not_publish_candidates(budgeted):
    x = s.Symbol("bracketed_stagnation")
    budget = s.EvaluationBudget() if budgeted else None
    assert float(s.nsolve(x-s.Rational(1, 2), x, (0, 1),
                         solver="bisect", maxiter=1, budget=budget)) == 0.5
    with pytest.raises(ValueError, match="failed to converge"):
        s.nsolve(x*x-2, x, (0, 2), solver="bisect", maxiter=1, budget=budget)
    left = 1.0
    right = math.nextafter(left, math.inf)
    # An unrepresentable root lies strictly between adjacent binary64 inputs.
    # Hold subtraction before scaling: expansion rounds the right endpoint's
    # residual to zero and would exercise the numerical-zero shortcut instead.
    source = s.Add(
        s.Mul(10**20, s.Add(x, -1, evaluate=False), evaluate=False),
        -10000,
        evaluate=False,
    )
    evaluate = (s.compile_numeric(x, source) if budgeted
                else s.lambdify(x, source, modules="math"))
    assert evaluate(left) < 0 < evaluate(right)
    with pytest.raises(ValueError, match="stagnat"):
        s.nsolve(source, x, (left, right), solver="bisect", tol=1e-30, budget=budget)


@pytest.mark.parametrize("bounds", [(), (0,), (0, 1, 2), (1, 0), (1, 1),
                                   (-math.inf, 1), (0, math.inf), (0, math.nan)])
def test_invalid_brackets_refuse_before_compilation(bounds, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid bounds must refuse before compilation")

    monkeypatch.setattr(numerical, "lambdify", forbidden)
    monkeypatch.setattr(numerical, "compile_numeric", forbidden)
    x = s.Symbol("bracketed_invalid_bounds")
    with pytest.raises(ValueError):
        s.nsolve(x-1, x, bounds, solver="bisect")


@pytest.mark.parametrize("option,value", [
    ("solver", "secant"), ("solver", 1), ("solver", lambda: None),
    ("tol", 0), ("tol", -1), ("tol", math.inf), ("tol", math.nan),
    ("maxiter", 0), ("maxiter", -1), ("maxiter", 10_001), ("maxiter", True),
    ("dict", 1), ("max_step_norm", 1), ("backend", "sparse"),
])
def test_invalid_controls_refuse_before_compilation(option, value, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid controls must refuse before compilation")

    monkeypatch.setattr(numerical, "lambdify", forbidden)
    monkeypatch.setattr(numerical, "compile_numeric", forbidden)
    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    x = s.Symbol("bracketed_invalid_controls")
    options = {"solver": "bisect", option: value}
    with pytest.raises((TypeError, ValueError)):
        s.nsolve(x-1, x, (0, 2), **options)


def test_bisection_rejects_systems_instead_of_treating_guesses_as_a_bracket(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar bisection cannot enter a system compiler")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    x, y = s.symbols("bracketed_system_x bracketed_system_y")
    for expression, variables in (([x-1], [x]), (x-1, [x]),
                                  ([x+y-1, x-y], [x, y]), (s.Matrix([x]), x)):
        with pytest.raises(ValueError, match="one scalar equation"):
            s.nsolve(expression, variables, (0, 2), solver="bisect")


def test_equal_signs_and_a_non_crossing_multiple_root_refuse():
    x = s.Symbol("bracketed_no_crossing")
    for source in (x*x+1, x*x, x+2):
        with pytest.raises(ValueError, match="opposite signs"):
            s.nsolve(source, x, (-1, 1), solver="bisect")


def test_unregistered_container_numeric_solver_and_function_hooks_are_not_called():
    calls = []

    class UserList(list):
        def __iter__(self):
            calls.append("iteration")
            return super().__iter__()

    class UserFloat:
        def __float__(self):
            calls.append("conversion")
            return 0.0

    class UserSolver(str):
        def __eq__(self, other):
            calls.append("solver comparison")
            return True

    class UserFunction(s.Function):
        def _eval_derivative(self, variable):
            calls.append("derivative")
            return s.Integer(1)

    x = s.Symbol("bracketed_closed_source")
    for bounds in (UserList([0, 2]), (UserFloat(), 2), ("0", 2), (1j, 2), 0):
        with pytest.raises(TypeError):
            s.nsolve(x-1, x, bounds, solver="bisect")
    with pytest.raises(TypeError):
        s.nsolve(x-1, x, (0, 2), solver=UserSolver("bisect"))
    with pytest.raises(NotImplementedError):
        s.nsolve(UserFunction(x, evaluate=False), x, (0, 2), solver="bisect")
    assert calls == []


def test_numerical_sources_and_typed_coordinates_are_not_printed(monkeypatch):
    x = s.Symbol("bracketed_print_independent")
    source = x*x-2

    def forbidden(*args, **kwargs):
        raise AssertionError("printers cannot provide numerical semantics")

    monkeypatch.setattr(s.Expr, "__str__", forbidden)
    monkeypatch.setattr(s.Symbol, "__str__", forbidden)
    assert abs(float(s.nsolve(source, x, (0, 2), solver="bisect"))**2-2) <= 1e-12


def test_one_cumulative_budget_covers_compilation_iterations_and_publication():
    x = s.Symbol("bracketed_cumulative_budget")
    source = x*x-2
    measured = ledger()
    root = s.nsolve(source, x, (0, 2), solver="bisect", budget=measured)
    assert abs(float(root)**2-2) <= 1e-12
    spent = LIMIT-measured.remaining_steps
    assert spent > 8
    assert 0 < measured.remaining_memory_bytes < LIMIT
    assert 0 < measured.remaining_allocations < LIMIT
    exact = ledger(max_steps=spent)
    assert float(s.nsolve(source, x, (0, 2), solver="bisect", budget=exact)) == float(root)
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(source, x, (0, 2), solver="bisect", budget=exact)
    published = []
    with pytest.raises(s.EvaluationBudgetExceeded):
        published.append(s.nsolve(source, x, (0, 2), solver="bisect",
                                  budget=ledger(max_steps=spent-1)))
    assert published == []


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_native_resource_failure_cannot_fall_back_to_unmetered_math(dimension, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")

    def forbidden(*args, **kwargs):
        raise AssertionError("budget refusal must not restart through another numerical lane")

    monkeypatch.setattr(numerical, "lambdify", forbidden)
    x = s.Symbol("bracketed_refused_budget")
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(x-1, x, (0, 2), solver="bisect", budget=ledger(**{dimension: 0}))


def test_unsupported_native_function_cannot_retry_the_math_backend(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("bracketed_unsupported_native")
    source = s.gamma(x)-2
    assert float(s.nsolve(source, x, (2, 4), solver="bisect")) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("native function refusal must not switch numerical backends")

    monkeypatch.setattr(numerical, "lambdify", forbidden)
    with pytest.raises((ValueError, NotImplementedError)):
        s.nsolve(source, x, (2, 4), solver="bisect", budget=ledger())


def test_cancelled_budget_refuses_before_source_compilation(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("bracketed_cancel_before")
    budget = ledger()
    budget.cancel()
    before = budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled work must not compile its source")

    monkeypatch.setattr(numerical, "compile_numeric", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        s.nsolve(x-1, x, (0, 2), solver="bisect", budget=budget)
    assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


def test_cancellation_after_an_endpoint_evaluation_prevents_publication(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    original = numerical.compile_numeric
    budget = ledger()
    evaluated = []

    def compile_then_cancel(*args, **kwargs):
        function = original(*args, **kwargs)

        def call(value, *, budget=None):
            result = function(value, budget=budget)
            evaluated.append(value)
            budget.cancel()
            return result

        return call

    monkeypatch.setattr(numerical, "compile_numeric", compile_then_cancel)
    x = s.Symbol("bracketed_cancel_endpoint")
    published = []
    with pytest.raises(s.EvaluationCancelled):
        published.append(s.nsolve(x, x, (0, 2), solver="bisect", budget=budget))
    assert evaluated == [0.0] and published == []
    assert budget.remaining_steps < LIMIT


def test_cancellation_at_result_materialization_prevents_mapping_publication(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    original = numerical.Float
    budget = ledger()
    x = s.Symbol("bracketed_cancel_publication")

    def cancel_after_float(value):
        result = original(value)
        budget.cancel()
        return result

    monkeypatch.setattr(numerical, "Float", cancel_after_float)
    published = []
    with pytest.raises(s.EvaluationCancelled):
        published.append(s.nsolve(x, x, (0, 2), solver="bisect", budget=budget, dict=True))
    assert published == []


@pytest.mark.parametrize("error_class", [ValueError, RuntimeError])
def test_compiler_failures_propagate_without_retry(error_class, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    failure = error_class("bracket compilation refused")
    calls = []

    def refused(*args, **kwargs):
        calls.append(1)
        raise failure

    def forbidden(*args, **kwargs):
        raise AssertionError("a failed native compilation cannot switch to math")

    monkeypatch.setattr(numerical, "compile_numeric", refused)
    monkeypatch.setattr(numerical, "lambdify", forbidden)
    x = s.Symbol("bracketed_compile_failure")
    with pytest.raises(error_class) as caught:
        s.nsolve(x-1, x, (0, 2), solver="bisect", budget=ledger())
    assert caught.value is failure and calls == [1]


def test_explicit_newton_selection_preserves_the_established_scalar_path():
    x = s.Symbol("bracketed_explicit_newton")
    assert s.nsolve(x*x-2, x, 1, solver="newton") == s.nsolve(x*x-2, x, 1)
    with pytest.raises(TypeError):
        s.nsolve(x*x-2, x, (0, 2))
