"""Actual-extension tests for sparse, approximate least-squares fitting.

All solves use the checked original residual program. Analytic linear and
nonlinear examples supply expected results; no numerical solver is mocked.
"""

import math

import pytest
import sympy as s


def test_sparse_fit_uses_every_row_and_reports_approximate_stationarity():
    x, y = s.symbols("sparse_fit_x sparse_fit_y")
    program = s.compile_residual_system([x - 1, y - 1, x + y - 4], (x, y))
    result = program.solve_least_squares([0, 0], backend="sparse")
    assert type(result) is s.NumericLeastSquaresCandidate
    assert result.point == pytest.approx((5 / 3, 5 / 3), abs=1e-9)
    assert result.residuals == pytest.approx((2 / 3, 2 / 3, -2 / 3), abs=1e-9)
    assert result.cost == pytest.approx(2 / 3, abs=1e-12)
    assert result.gradient_norm <= 1e-10
    assert result.iterations > 0 and result.evaluations >= result.iterations + 2
    assert result.termination == "gradient_tolerance"
    assert result.backend == "sparse_lsqr" and result.evidence == "approximate"
    assert result.variables[0] is x and result.variables[1] is y
    assert not hasattr(result, "root") and not hasattr(result, "certificate")

    residuals, jacobian = program.evaluate(*result.point)
    assert result.residuals == tuple(residuals)
    gradient = [
        sum(row[column] * value for row, value in zip(jacobian, residuals))
        for column in range(2)
    ]
    assert result.gradient_norm == pytest.approx(max(map(abs, gradient)), abs=1e-15)
    native = program.native_system.solve_least_squares([0.0, 0.0], backend="sparse")
    assert native[0] == pytest.approx(result.point, abs=1e-12)
    assert native[1] == pytest.approx(result.residuals, abs=1e-12)
    assert native[2:4] == pytest.approx((result.cost, result.gradient_norm), abs=1e-12)
    assert native[-1] == "gradient_tolerance"

    with pytest.raises(AttributeError):
        result.backend = "dense"
    with pytest.raises(AttributeError):
        del result._backend


def test_default_dense_backend_keeps_its_existing_result_and_controls():
    x = s.Symbol("sparse_default_x")
    program = s.compile_residual_system([x - 1, x - 3], x)
    implicit = program.solve_least_squares(0)
    explicit = program.solve_least_squares(0, backend="dense")
    assert implicit.point == explicit.point
    assert implicit.residuals == explicit.residuals
    assert implicit.cost == explicit.cost
    assert implicit.iterations == explicit.iterations
    assert implicit.evaluations == explicit.evaluations
    assert implicit.backend == explicit.backend == "frankennumpy_qr"
    assert implicit.point == pytest.approx((2,), abs=1e-12)
    assert implicit.cost == pytest.approx(1, abs=1e-12)


def test_rank_deficient_fit_requires_explicit_sparse_selection():
    x, y = s.symbols("sparse_rank_x sparse_rank_y")
    program = s.compile_residual_system(
        [x + y - 1, 2*x + 2*y - 2, 3*x + 3*y - 3], (x, y),
    )
    for callable_ in (program, program.native_system):
        with pytest.raises(ValueError, match="rank deficient"):
            callable_.solve_least_squares([0, 0])
    result = program.solve_least_squares([0, 0], backend="sparse")
    assert sum(result.point) == pytest.approx(1, abs=1e-10)
    assert max(map(abs, result.residuals)) <= 1e-9
    assert result.gradient_norm <= 1e-10
    assert result.evidence == "approximate"


def test_underdetermined_fit_preserves_unused_coordinates_without_a_uniqueness_claim():
    x, y, unused = s.symbols("sparse_under_x sparse_under_y sparse_unused")
    program = s.compile_residual_system([x + 2*y - 4], (x, y, unused))
    for callable_ in (program, program.native_system):
        with pytest.raises(ValueError, match="least squares requires"):
            callable_.solve_least_squares([0, 0, 7])
    result = program.solve_least_squares([0, 0, 7], backend="sparse")
    assert result.point[0] + 2*result.point[1] == pytest.approx(4, abs=1e-10)
    assert result.point[2] == 7
    assert result.gradient_norm <= 1e-10
    assert result.cost < 1e-18
    assert result.backend == "sparse_lsqr" and result.evidence == "approximate"
    assert not hasattr(result, "certificate")


def test_sparse_blocks_extend_both_dense_shape_limits():
    variables = s.symbols("sparse_block_0:24")
    targets = tuple((index % 7) - 3 for index in range(len(variables)))
    residuals = [
        weight * (variable - target)
        for weight in (1, 2, 3)
        for variable, target in zip(variables, targets)
    ]
    program = s.compile_residual_system(residuals, variables)
    assert len(program.jacobian_pattern) == 72
    for callable_ in (program, program.native_system):
        with pytest.raises(ValueError, match="least squares requires"):
            callable_.solve_least_squares([0] * len(variables))
    result = program.solve_least_squares([0] * len(variables), backend="sparse")
    assert result.point == pytest.approx(targets, abs=1e-9)
    assert len(result.residuals) == 72
    assert result.gradient_norm <= 1e-10
    assert result.cost < 1e-17


def test_column_scaling_does_not_change_the_unit_weighted_objective():
    x, y = s.symbols("sparse_scaled_x sparse_scaled_y")
    program = s.compile_residual_system(
        [10**6*x - 1, y - 2, 10**6*x + y - 5], (x, y),
    )
    result = program.solve_least_squares([0, 0], backend="sparse", gtol=1e-8)
    assert result.point == pytest.approx((5 / (3 * 10**6), 8 / 3), rel=1e-9, abs=1e-12)
    assert result.residuals == pytest.approx((2 / 3, 2 / 3, -2 / 3), abs=1e-9)
    assert result.cost == pytest.approx(2 / 3, abs=1e-12)
    assert result.gradient_norm <= 1e-8


def test_nonlinear_inconsistent_fit_uses_the_original_cost_gradient():
    x = s.Symbol("sparse_nonlinear_x")
    program = s.compile_residual_system([x**2 - 2, x - 1], x)
    result = program.solve_least_squares(1, backend="sparse")
    expected = (1 + math.sqrt(3)) / 2
    assert result.point == pytest.approx((expected,), abs=1e-8)
    point = result.point[0]
    residuals = (point**2 - 2, point - 1)
    assert result.residuals == pytest.approx(residuals, abs=1e-14)
    assert result.cost == pytest.approx(sum(value*value for value in residuals) / 2)
    assert abs(2*point*residuals[0] + residuals[1]) <= 1e-10
    assert result.cost > 0 and result.gradient_norm <= 1e-10


def test_snapshot_transport_and_typed_coordinate_order_survive_sparse_selection():
    x = s.Symbol("sparse_same_name", real=True)
    y = s.Symbol("sparse_same_name", positive=True)
    dummy = s.Dummy("sparse_same_name")
    source = s.SparseMatrix([x - 1, y - 2, dummy - 3])
    expected = source.as_immutable()
    layout = [(dummy, x), y]
    program = s.compile_residual_system(source, layout)
    data = program.to_bytes()
    source[0, 0] = 999
    layout[0] = (x, dummy)
    restored = s.load_residual_system(data, residuals=expected, args=[(dummy, x), y])
    for candidate_program in (program, restored):
        result = candidate_program.solve_least_squares([0, 0, 0], backend="sparse")
        assert result.point == pytest.approx((3, 1, 2), abs=1e-9)
        assert result.variables[0] is dummy
        assert result.variables[1] is x
        assert result.variables[2] is y
        assert max(map(abs, result.residuals)) <= 1e-9
        assert result.backend == "sparse_lsqr"


@pytest.mark.parametrize("kind,initial,expected", [
    ("log", 10, 1), ("sqrt", 4, 1), ("exp", -7, 0),
])
def test_invalid_sparse_trials_backtrack_with_original_domains(kind, initial, expected):
    x = s.Symbol("sparse_trial_x")
    residual = {"log": s.log(x), "sqrt": s.sqrt(x) - 1, "exp": s.exp(x) - 1}[kind]
    program = s.compile_residual_system([residual, 2*residual], x)
    result = program.solve_least_squares(initial, backend="sparse")
    assert result.point == pytest.approx((expected,), abs=1e-8)
    assert result.evaluations > result.iterations + 2
    assert result.gradient_norm <= 1e-10 and result.cost < 1e-17
    assert all(math.isfinite(value) for value in result.point + result.residuals)


@pytest.mark.parametrize("kind,initial", [
    ("log", 0), ("log", -1), ("sqrt", 0), ("exp", 550),
])
def test_invalid_initial_source_or_derivative_is_terminal(kind, initial):
    x = s.Symbol("sparse_initial_domain")
    residual = {"log": s.log(x), "sqrt": s.sqrt(x), "exp": s.exp(x) - 1}[kind]
    program = s.compile_residual_system([residual, 2*residual], x)
    for callable_, point in ((program, initial), (program.native_system, [float(initial)])):
        budget = s.EvaluationBudget()
        before = budget.remaining_steps
        with pytest.raises(ValueError):
            callable_.solve_least_squares(point, backend="sparse", budget=budget)
        assert budget.remaining_steps < before


@pytest.mark.parametrize("hidden_kind", ["zero_product", "zero_power"])
def test_zero_residuals_cannot_hide_original_source_holes(hidden_kind):
    x = s.Symbol("sparse_retained_domain")
    logarithm = s.log(x)
    hidden = (s.Mul(0, logarithm, evaluate=False) if hidden_kind == "zero_product"
              else s.Add(s.Pow(logarithm, 0, evaluate=False), -1, evaluate=False))
    program = s.compile_residual_system([x, hidden], x)
    for callable_, point in ((program, 0), (program.native_system, [0.0])):
        with pytest.raises(ValueError):
            callable_.solve_least_squares(point, backend="sparse")


@pytest.mark.parametrize("constant", [False, True])
def test_stationary_nonroots_are_not_promoted_to_minima_or_root_certificates(constant):
    x = s.Symbol("sparse_stationary_nonroot")
    program = s.compile_residual_system([1 if constant else x**2 - 1], x)
    result = program.solve_least_squares(0, backend="sparse", maxiter=0)
    assert result.point == (0.0,)
    assert result.residuals == ((1.0,) if constant else (-1.0,))
    assert result.cost == 0.5 and result.gradient_norm == 0
    assert result.iterations == 0 and result.evaluations == 2
    assert result.evidence == "approximate" and not hasattr(result, "certificate")


def test_zero_iterations_or_vanishing_steps_cannot_report_convergence():
    x = s.Symbol("sparse_stopping_x")
    program = s.compile_residual_system([x - 1, 2*x - 2], x)
    with pytest.raises(ValueError, match="stationarity"):
        program.solve_least_squares(0, backend="sparse", maxiter=0)
    assert program.solve_least_squares(1, backend="sparse", maxiter=0).iterations == 0
    with pytest.raises(ValueError, match="stationarity"):
        program.solve_least_squares(
            2, backend="sparse", gtol=0,
            max_step_norm=float.fromhex("0x0.0000000000001p-1022"),
        )


@pytest.mark.parametrize("limit,remaining", [
    ("max_steps", "remaining_steps"),
    ("max_memory_bytes", "remaining_memory_bytes"),
    ("max_allocations", "remaining_allocations"),
])
def test_sparse_work_and_rejected_trials_use_one_cumulative_budget(limit, remaining):
    x = s.Symbol("sparse_budget_x")
    program = s.compile_residual_system([s.log(x), 2*s.log(x)], x)
    measured = s.EvaluationBudget()
    before = getattr(measured, remaining)
    result = program.solve_least_squares(10, backend="sparse", budget=measured)
    cost = before - getattr(measured, remaining)
    assert cost > 0 and result.evaluations > result.iterations + 2

    exact = s.EvaluationBudget(**{limit: cost})
    assert program.solve_least_squares(10, backend="sparse", budget=exact).point == result.point
    assert getattr(exact, remaining) == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_least_squares(10, backend="sparse", budget=exact)
    short = s.EvaluationBudget(**{limit: cost - 1})
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_least_squares(10, backend="sparse", budget=short)


def test_cancelled_sparse_calls_return_no_candidate_at_either_entrypoint():
    x = s.Symbol("sparse_cancel_x")
    program = s.compile_residual_system([x - 1], x)
    for callable_, initial in ((program, 0), (program.native_system, [0.0])):
        cancelled = s.EvaluationBudget()
        cancelled.cancel()
        with pytest.raises(s.EvaluationCancelled):
            callable_.solve_least_squares(initial, backend="sparse", budget=cancelled)


@pytest.mark.parametrize("options", [
    {"backend": "auto"}, {"backend": "frankennumpy"}, {"backend": "SPARSE"},
    {"backend": "sparse", "gtol": -1}, {"backend": "sparse", "gtol": math.nan},
    {"backend": "sparse", "maxiter": -1}, {"backend": "sparse", "maxiter": 1025},
    {"backend": "sparse", "max_step_norm": 0},
    {"backend": "sparse", "max_step_norm": math.inf},
])
def test_invalid_controls_refuse_before_input_hooks_or_native_work(options):
    x = s.Symbol("sparse_invalid_option")
    program = s.compile_residual_system([x - 1], x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("invalid options ran an input conversion hook")

    for callable_, initial in ((program, Initial()), (program.native_system, [Initial()])):
        budget = s.EvaluationBudget()
        before = budget.remaining_steps
        with pytest.raises(ValueError):
            callable_.solve_least_squares(initial, budget=budget, **options)
        assert effects == [] and budget.remaining_steps == before


@pytest.mark.parametrize("bad", [None, 1, b"sparse"])
def test_backend_selector_requires_a_string_at_both_entrypoints(bad):
    x = s.Symbol("sparse_backend_type")
    program = s.compile_residual_system([x - 1], x)
    for callable_, initial in ((program, 0), (program.native_system, [0.0])):
        with pytest.raises(TypeError):
            callable_.solve_least_squares(initial, backend=bad)


@pytest.mark.parametrize("case", ["variables", "residuals", "differentiation_work"])
def test_sparse_selection_does_not_relax_existing_compilation_caps(case):
    if case == "variables":
        variables = s.symbols("sparse_cap_0:257")
        residuals = [variables[0]]
    elif case == "residuals":
        variables = (s.Symbol("sparse_cap_rows"),)
        residuals = [variables[0]] * 4097
    else:
        variables = s.symbols("sparse_cap_0:256")
        residuals = [s.Add(*variables, evaluate=False)]
    with pytest.raises(ValueError):
        s.compile_residual_system(residuals, variables)


def test_undefined_intrinsic_spelling_has_no_closed_solver_capability():
    x = s.Symbol("sparse_opaque_x")
    with pytest.raises(NotImplementedError):
        s.compile_residual_system([s.Function("sin")(x)], x)
