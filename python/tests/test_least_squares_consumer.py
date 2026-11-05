"""Actual-extension regressions for the checked rectangular fitting consumer."""

import math

import pytest
import sympy as s
from sympy.utilities import NumericLeastSquaresCandidate


def test_every_residual_row_contributes_to_the_inconsistent_hero_fit():
    x, y = s.symbols("least_squares_x least_squares_y")
    program = s.compile_residual_system([x - 1, y - 1, x + y - 4], (x, y))
    result = program.solve_least_squares([0, 0])
    assert type(result) is NumericLeastSquaresCandidate is s.NumericLeastSquaresCandidate
    assert result.point == pytest.approx((5 / 3, 5 / 3), abs=1e-12)
    assert result.residuals == pytest.approx((2 / 3, 2 / 3, -2 / 3), abs=1e-12)
    assert result.cost == pytest.approx(2 / 3, abs=1e-12)
    assert result.gradient_norm <= 1e-10
    assert result.iterations > 0 and result.evaluations >= result.iterations + 2
    assert result.termination == "gradient_tolerance"
    assert result.backend == "frankennumpy_qr" and result.evidence == "approximate"
    assert result.variables[0] is x and result.variables[1] is y
    assert not hasattr(result, "root") and not hasattr(result, "certificate")
    residuals, jacobian = program.evaluate(*result.point)
    assert tuple(residuals) == result.residuals
    gradient = [sum(row[j] * value for row, value in zip(jacobian, residuals)) for j in range(2)]
    assert max(map(abs, gradient)) == pytest.approx(result.gradient_norm, abs=1e-15)


def test_nonlinear_inconsistent_fit_has_the_expected_stationary_point():
    x = s.Symbol("least_squares_nonlinear")
    program = s.compile_residual_system([x**2 - 2, x - 1], x)
    result = program.solve_least_squares(1)
    assert result.point == pytest.approx(((1 + math.sqrt(3)) / 2,), abs=1e-8)
    assert result.cost > 0 and result.gradient_norm <= 1e-10
    assert result.residuals == pytest.approx(
        (result.point[0]**2 - 2, result.point[0] - 1), abs=1e-14,
    )


def test_stationarity_at_a_local_maximum_does_not_become_a_root_or_minimum_claim():
    x = s.Symbol("least_squares_nonminimum")
    program = s.compile_residual_system([x**2 - 1], x)
    result = program.solve_least_squares(0, maxiter=0)
    assert result.point == (0.0,) and result.residuals == (-1.0,)
    assert result.cost == 0.5 and result.gradient_norm == 0
    assert result.iterations == 0 and result.evaluations == 2
    assert result.evidence == "approximate"
    with pytest.raises(ValueError):
        program.solve_candidate(0, maxiter=0)


def test_rank_deficient_nonstationary_system_refuses_without_resetting_budget():
    x, y = s.symbols("least_squares_rank_x least_squares_rank_y")
    program = s.compile_residual_system(
        [x + y - 1, 2*x + 2*y - 2, 3*x + 3*y - 3], (x, y),
    )
    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    with pytest.raises(ValueError, match="rank deficient"):
        program.solve_least_squares([0, 0], budget=budget)
    assert budget.remaining_steps < before
    result = program.solve_least_squares([1, 0], budget=budget)
    assert result.cost == 0 and result.iterations == 0


@pytest.mark.parametrize("hidden_kind", ["zero_product", "zero_power"])
def test_a_trailing_held_domain_hole_is_checked_even_at_zero_leading_rows(hidden_kind):
    x, y = s.symbols("least_squares_domain_x least_squares_domain_y")
    logarithm = s.log(x)
    hidden = (s.Mul(0, logarithm, evaluate=False) if hidden_kind == "zero_product"
              else s.Add(s.Pow(logarithm, 0, evaluate=False), -1, evaluate=False))
    program = s.compile_residual_system([x, y, hidden], (x, y))
    for call in (
        lambda: program.solve_least_squares([0, 0]),
        lambda: program.native_system.solve_least_squares([0.0, 0.0]),
    ):
        with pytest.raises(ValueError):
            call()


def test_derivative_domain_is_checked_before_a_zero_residual_can_finish():
    x = s.Symbol("least_squares_derivative_domain")
    program = s.compile_residual_system([s.sqrt(x), x], x)
    assert program.residuals(0) == [0, 0]
    with pytest.raises(ValueError):
        program.solve_least_squares(0)


@pytest.mark.parametrize("source", ["10**308", "10**154"])
def test_finite_residuals_do_not_authorize_an_overflowing_cost(source):
    x = s.Symbol("least_squares_cost")
    constant = s.Integer(10)**(308 if source == "10**308" else 154)
    program = s.compile_residual_system([constant, constant], x)
    with pytest.raises(ValueError, match="squared residual cost"):
        program.solve_least_squares(0)


def test_original_matrix_source_nested_layout_and_transport_stay_bound():
    x, y = s.symbols("least_squares_transport_x least_squares_transport_y")
    source = s.SparseMatrix([x - 1, y - 1, x + y - 4])
    expected = source.as_immutable()
    layout = [(y, x)]
    program = s.compile_residual_system(source, layout)
    encoded = program.to_bytes()
    source[0, 0] = 999
    layout[0] = (x, y)
    restored = s.load_residual_system(encoded, residuals=expected, args=[(y, x)])
    for candidate_program in (program, restored):
        result = candidate_program.solve_least_squares([0, 0])
        assert result.point == pytest.approx((5 / 3, 5 / 3), abs=1e-12)
        assert result.variables[0] is y and result.variables[1] is x
        assert tuple(candidate_program.residuals(result.point)) == result.residuals
    assert program.to_bytes() == restored.to_bytes() == encoded


@pytest.mark.parametrize("field", ["point", "residuals", "cost", "gradient_norm", "iterations"])
def test_result_coordinates_and_diagnostics_are_immutable(field):
    x = s.Symbol("least_squares_immutable")
    result = s.compile_residual_system([x - 1, 2*x - 2], x).solve_least_squares(0)
    with pytest.raises(AttributeError):
        setattr(result, field, None)
    with pytest.raises(AttributeError):
        delattr(result, field)
    with pytest.raises(AttributeError):
        result.__init__((0,), (x,), (1, 2), 2.5, 5, 1, 2, "gradient_tolerance")


def test_native_bridge_exposes_the_full_rectangular_result():
    x, y = s.symbols("least_squares_bridge_x least_squares_bridge_y")
    program = s.compile_residual_system([x - 1, y - 1, x + y - 4], (x, y))
    point, residuals, cost, gradient, iterations, evaluations, reason = (
        program.native_system.solve_least_squares([0.0, 0.0])
    )
    assert point == pytest.approx([5 / 3, 5 / 3], abs=1e-12)
    assert residuals == pytest.approx([2 / 3, 2 / 3, -2 / 3], abs=1e-12)
    assert cost == pytest.approx(2 / 3, abs=1e-12)
    assert gradient <= 1e-10 and evaluations >= iterations + 2
    assert reason == "gradient_tolerance"


@pytest.mark.parametrize("options", [
    {"gtol": -1}, {"gtol": float("nan")}, {"maxiter": -1}, {"maxiter": 1025},
    {"max_step_norm": 0}, {"max_step_norm": float("inf")},
])
def test_invalid_controls_refuse_before_initial_conversion(options):
    x = s.Symbol("least_squares_controls")
    program = s.compile_residual_system([x - 1, 2*x - 2], x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("invalid options must not invoke conversion hooks")

    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    for callable_, initial in (
        (program, Initial()), (program.native_system, [Initial()]),
    ):
        with pytest.raises(ValueError):
            callable_.solve_least_squares(initial, budget=budget, **options)
    assert effects == [] and budget.remaining_steps == before


def test_zero_iterations_and_stagnation_do_not_report_false_convergence():
    x = s.Symbol("least_squares_stop")
    program = s.compile_residual_system([x - 1, 2*x - 2], x)
    with pytest.raises(ValueError, match="stationarity"):
        program.solve_least_squares(0, maxiter=0)
    assert program.solve_least_squares(1, maxiter=0).iterations == 0
    with pytest.raises(ValueError, match="stationarity"):
        program.solve_least_squares(2, gtol=0, max_step_norm=float.fromhex("0x0.0000000000001p-1022"))


def test_shared_budget_exhaustion_and_cancellation_remain_typed():
    x = s.Symbol("least_squares_budget")
    program = s.compile_residual_system([x - 1, 2*x - 2], x)
    measured = s.EvaluationBudget()
    initial = measured.remaining_steps
    program.solve_least_squares(0, budget=measured)
    cost = initial - measured.remaining_steps
    assert cost > 0
    shared = s.EvaluationBudget(max_steps=cost)
    program.solve_least_squares(0, budget=shared)
    assert shared.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_least_squares(0, budget=shared)
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.solve_least_squares(0, budget=cancelled)


@pytest.mark.parametrize("shape", ["underdetermined", "too_many_variables", "too_many_rows"])
def test_unsupported_shapes_refuse_at_both_entrypoints(shape):
    if shape == "too_many_variables":
        variables = s.symbols("ls0:17")
        residuals = list(variables)
    else:
        x, y = s.symbols("least_squares_shape_x least_squares_shape_y")
        variables = (x, y) if shape == "underdetermined" else (x,)
        residuals = [x] if shape == "underdetermined" else [x] * 65
    program = s.compile_residual_system(residuals, variables)
    initial = [0] * len(variables)
    for callable_ in (program, program.native_system):
        with pytest.raises(ValueError, match="least squares requires"):
            callable_.solve_least_squares(initial)
