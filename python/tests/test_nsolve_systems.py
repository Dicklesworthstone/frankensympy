"""Real residual systems must use the checked Jacobian and satisfy every row."""

import importlib
import math

import pytest

from sympy import (
    Add, Dummy, Eq, Expr, Float, Function, ImmutableMatrix, Matrix, Rational,
    SparseMatrix, Symbol, exp, log, nsolve, sin, sqrt,
)


def values(result):
    assert type(result) is Matrix
    assert result.cols == 1
    assert all(type(result[row, 0]) is Float for row in range(result.rows))
    return [float(result[row, 0]) for row in range(result.rows)]


def test_nonlinear_square_system_solves_the_pinned_upstream_example():
    x, y = Symbol("x"), Symbol("y")
    result = values(nsolve(
        (3*x**2 - 2*y**2 - 1, x**2 - 2*x + y**2 + 2*y - 8),
        (x, y), (-1, 1),
    ))
    a, b = result
    assert result == pytest.approx([-1.1928730993524608, 1.278444111699106], abs=1e-12)
    assert abs(3*a*a - 2*b*b - 1) <= 1e-12
    assert abs(a*a - 2*a + b*b + 2*b - 8) <= 1e-12


def test_transcendental_system_uses_symbolic_derivatives():
    x, y = Symbol("x"), Symbol("y")
    a, b = values(nsolve((exp(x) - 2, x + y - 1), (x, y), (0, 0)))
    assert a == pytest.approx(math.log(2), abs=1e-12)
    assert b == pytest.approx(1 - math.log(2), abs=1e-12)
    assert abs(math.exp(a) - 2) <= 1e-12
    assert abs(a + b - 1) <= 1e-12


def test_backtracking_recovers_from_a_real_domain_crossing():
    x, y = Symbol("x"), Symbol("y")
    # The undamped Newton trial for log(x) at x=10 is negative.
    assert values(nsolve((log(x), y - 1), (x, y), (10, 0))) == pytest.approx([1, 1])


def test_backtracking_recovers_from_an_oversized_nonlinear_step():
    x, y = Symbol("x"), Symbol("y")
    assert values(nsolve((x**3 - 1, y - 2), (x, y), (0.1, 0))) == pytest.approx([1, 2])


def test_overdetermined_consistent_system_checks_all_equations():
    x, y = Symbol("x"), Symbol("y")
    result = values(nsolve((x + y - 3, 2*x - y, 3*x - 3), (x, y), (0, 0)))
    assert result == pytest.approx([1, 2], abs=1e-12)
    a, b = result
    assert max(abs(a+b-3), abs(2*a-b), abs(3*a-3)) <= 1e-12


def test_scaled_equations_and_variables_preserve_small_sensitivities():
    x, y = Symbol("x"), Symbol("y")
    assert values(nsolve(
        (10**200*(x+y-3), (x-y+1)/10**200, x+2*y-5), (x, y), (0, 0),
    )) == pytest.approx([1, 2], abs=1e-12)
    a, b = values(nsolve(
        (10**100*x+y-3, 10**100*x-y+1), (x, y), (0, 0),
    ))
    assert a == pytest.approx(1e-100, rel=1e-12, abs=0)
    assert b == pytest.approx(2, abs=1e-12)


def test_subnormal_derivatives_work_and_unrepresentable_steps_refuse():
    x, y = Symbol("x"), Symbol("y")
    assert values(nsolve(((x-2)/10**308, y-2), (x, y), (0, 0))) \
        == pytest.approx([2, 2])
    with pytest.raises(ValueError, match="Newton step"):
        nsolve((x/10**308+2, y), (x, y), (1e308, 0))


def test_pivoted_qr_handles_a_nearly_dependent_overdetermined_system():
    x, y = Symbol("x"), Symbol("y")
    epsilon = Rational(1, 10**8)
    a, b = values(nsolve((
        x+y-3,
        x+(1+epsilon)*y-(3+2*epsilon),
        x+(1-epsilon)*y-(3-2*epsilon),
    ), (x, y), (0, 0)))
    # Forming J.T*J loses the epsilon**2 separation in binary64.
    assert (a, b) == pytest.approx((1, 2), abs=1e-7)
    assert abs(a+b-3) <= 1e-12


@pytest.mark.parametrize("matrix_type", [Matrix, ImmutableMatrix, SparseMatrix])
def test_matrix_vector_inputs_are_typed_and_not_mutated(matrix_type):
    x, y = Symbol("x"), Symbol("y")
    equations = matrix_type([x+y-3, x-y+1])
    variables = matrix_type([[x, y]])
    initial = matrix_type([0, 0])
    before = equations[0, 0], equations[1, 0]
    assert values(nsolve(equations, variables, initial)) == pytest.approx([1, 2])
    assert (equations[0, 0], equations[1, 0]) == before
    assert float(initial[0, 0]) == float(initial[1, 0]) == 0


def test_scalar_and_system_equalities_and_solution_mappings():
    x, y = Symbol("x"), Symbol("y")
    scalar = nsolve(Eq(x**2, 2), x, 1, dict=True)
    assert type(scalar) is list and len(scalar) == 1
    assert list(scalar[0]) == [x]
    assert type(scalar[0][x]) is Float
    assert float(scalar[0][x]) == pytest.approx(math.sqrt(2), abs=1e-12)
    mapped = nsolve((Eq(x+y, 3), Eq(x-y, -1)), (x, y), (0, 0), dict=True)
    assert list(mapped[0]) == [x, y]
    assert [float(mapped[0][symbol]) for symbol in (x, y)] == pytest.approx([1, 2])
    assert type(nsolve(Eq(x, 2), x, 0)) is Float


def test_same_printed_names_and_dummies_keep_distinct_slots():
    ordinary, assumed = Symbol("x"), Symbol("x", positive=True)
    first, second = Dummy("same"), Dummy("same")
    result = nsolve((ordinary+assumed-3, ordinary-assumed+1,
                     first+second-7, first-second+1),
                    (ordinary, assumed, first, second), (0, 0, 0, 0), dict=True)
    assert len(result[0]) == 4
    assert [float(result[0][symbol]) for symbol in (ordinary, assumed, first, second)] \
        == pytest.approx([1, 2, 3, 4])


def test_single_variable_system_still_returns_a_column_matrix():
    x = Symbol("x")
    assert values(nsolve([x-2], [x], [0])) == [2.0]
    assert values(nsolve([x-2, 2*x-4], [x], 0)) == pytest.approx([2.0])


def test_initial_root_does_not_evaluate_a_singular_jacobian():
    x, y = Symbol("x"), Symbol("y")
    assert values(nsolve((sqrt(x), y**2), (x, y), (0, 0))) == [0.0, 0.0]
    assert values(nsolve((x+y, 2*x+2*y), (x, y), (0, 0))) == [0.0, 0.0]


def test_fractional_power_systems_compile_checked_jacobians_and_keep_real_branches():
    x, y = Symbol("x"), Symbol("y")
    assert values(nsolve((sqrt(x)-2, y-x), (x, y), (1, 1))) \
        == pytest.approx([4, 4], abs=1e-12)
    assert values(nsolve((sqrt(x**2)+y-3, y-1), (x, y), (-1, 0))) \
        == pytest.approx([-2, 1], abs=1e-12)
    with pytest.raises(ValueError):
        nsolve((sqrt(x)-2, y), (x, y), (-1, 0))


def test_inconsistent_system_never_returns_a_least_squares_minimizer_as_a_root():
    x = Symbol("x")
    with pytest.raises(ValueError):
        nsolve((x-1, x-2), (x,), (0,))


def test_rank_deficiency_and_tiny_steps_with_large_residuals_refuse():
    x, y = Symbol("x"), Symbol("y")
    with pytest.raises(ValueError, match="rank-deficient"):
        nsolve((x+y-1, 2*x+2*y-3), (x, y), (0, 0))
    with pytest.raises(ValueError):
        nsolve((1+(10**30*x)**2, y), (x, y), (1e-30, 0))
    with pytest.raises(ValueError):
        nsolve((x**2+1, y), (x, y), (1, 0))


def test_system_stagnation_and_exhausted_iterations_refuse():
    x, y = Symbol("x"), Symbol("y")
    with pytest.raises(ValueError, match="stagnat"):
        nsolve((exp(10**20*(x-1))+1, y), (x, y), (1, 0))
    with pytest.raises(ValueError, match="failed to converge"):
        nsolve((x**2-2, y), (x, y), (1, 0), maxiter=1)


def test_nonfinite_initial_values_and_initial_domain_errors_refuse():
    x, y = Symbol("x"), Symbol("y")
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            nsolve((x-1, y-1), (x, y), (value, 0))
    with pytest.raises(ValueError):
        nsolve((log(x), y-1), (x, y), (-1, 0))
    with pytest.raises(ValueError):
        nsolve((exp(x), y), (x, y), (1000, 0))


def test_dimension_and_shape_checks_run_before_native_compilation(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x, y = Symbol("x"), Symbol("y")

    def forbidden(*args, **kwargs):
        raise AssertionError("shape-invalid systems must refuse before compilation")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    for expressions, variables, initial in (
        ([], [x], [0]), ([x], [], []), ([x], [x, y], [0, 0]),
        ([x], x, [0]), (x, [x], [0]),
        ([x, y], [x, x], [0, 0]), ([x, y], [x, y], [0]),
        (Matrix([[x, y], [x, y]]), [x, y], [0, 0]),
        ([x]*257, [x], [0]),
        ([x]*65, [Symbol(f"v{i}") for i in range(65)], [0]*65),
        ([x]*65, [Symbol(f"v{i}") for i in range(64)], [0]*64),
    ):
        with pytest.raises((ValueError, TypeError)):
            nsolve(expressions, variables, initial)


def test_evaluation_and_dense_work_budgets_are_not_reset_during_iteration(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("x")
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 2)
    with pytest.raises(ValueError, match="evaluation budget"):
        nsolve([x**2-2], [x], [1])
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 4096)
    monkeypatch.setattr(numerical, "_MAX_LINEAR_WORK", 0)
    with pytest.raises(ValueError, match="linear-algebra work budget"):
        nsolve([x**2-2], [x], [1])


def test_no_expression_printing_is_used_to_bind_or_solve_systems(monkeypatch):
    x, y = Symbol("not an identifier"), Symbol("sin")
    expressions = [x+y-3, x-y+1]

    def forbidden(*args, **kwargs):
        raise AssertionError("printers cannot supply numerical semantics")

    monkeypatch.setattr(Expr, "__str__", forbidden)
    monkeypatch.setattr(Symbol, "__str__", forbidden)
    assert values(nsolve(expressions, (x, y), (0, 0))) == pytest.approx([1, 2])


def test_custom_functions_containers_and_numeric_hooks_are_not_executed():
    calls = []
    x = Symbol("x")

    class UserFunction(Function):
        def _eval_derivative(self, variable):
            calls.append("derivative")
            return 1

    class UserList(list):
        def __iter__(self):
            calls.append("iteration")
            return super().__iter__()

    class UserFloat:
        def __float__(self):
            calls.append("numeric conversion")
            return 0.0

    class UserSymbol(Symbol):
        pass

    for expression, variables, initial in (
        ([UserFunction(x, evaluate=False)], [x], [0]),
        (UserList([x-1]), [x], [0]),
        ([x-1], [x], [UserFloat()]),
        ([x-1], [UserSymbol("x")], [0]),
    ):
        with pytest.raises((TypeError, NotImplementedError)):
            nsolve(expression, variables, initial)
    assert calls == []


def test_held_builtin_equations_use_the_original_residual_expression():
    x, y = Symbol("x"), Symbol("y")
    held = Add(x, x, evaluate=False)
    before = held.args
    assert values(nsolve((held+y-4, x-y+1), (x, y), (0, 0))) \
        == pytest.approx([1, 2])
    assert held.args == before


def test_unbound_symbols_and_non_boolean_mapping_option_refuse():
    x, y = Symbol("x"), Symbol("y")
    with pytest.raises(ValueError):
        nsolve([x+y], [x], [0])
    with pytest.raises(TypeError):
        nsolve([x-1], [x], [0], dict=1)

