"""Actual-extension coverage of source-bound box-constrained sparse fitting."""

import importlib
import math

import pytest
import sympy as s


def scalar_program(residual=None):
    x = s.Symbol("bounded_fit_x")
    return s.compile_residual_system([x - 2 if residual is None else residual(x)], x), x


def test_boundary_stationarity_reports_the_full_and_projected_gradients_separately():
    program, x = scalar_program()
    result = program.solve_bounded_least_squares(0, [(0, 1)])
    assert type(result) is s.NumericBoundedLeastSquaresCandidate
    assert result.native_system is program.native_system
    assert result.variables == (x,) and result.variables[0] is x
    assert result.point == (1.0,)
    assert result.bounds == ((0.0, 1.0),)
    assert result.residuals == (-1.0,)
    assert result.cost == 0.5
    assert result.gradient_norm == 1.0
    assert result.projected_gradient_norm == 0.0
    assert result.backend == "sparse_bounded_lsqr"
    assert result.termination == "projected_gradient_tolerance"
    assert result.evidence == "approximate"
    assert not hasattr(result, "certificate")
    assert result.evaluations >= result.iterations + 2


@pytest.mark.parametrize("second,expected", [
    (-1, (0.0, -0.2)),
    (1, (1.0, -1.0)),
])
def test_coupled_fit_refits_and_releases_active_coordinates(second, expected):
    x, y = s.symbols("bounded_coupled_x bounded_coupled_y")
    program = s.compile_residual_system([x + 2*y + 1, y + second], (x, y))
    result = program.solve_bounded_least_squares([0, 0], [(0, None), (None, None)])
    assert result.point == pytest.approx(expected, abs=1e-9)
    assert result.projected_gradient_norm <= 1e-10
    first, second_value = result.point
    expected_residuals = (first + 2*second_value + 1, second_value + second)
    assert result.residuals == pytest.approx(expected_residuals, abs=1e-12)
    assert result.cost == pytest.approx(sum(value*value for value in expected_residuals) / 2)


def test_near_boundary_coupling_can_use_projected_gradient_after_damped_trials():
    x, y = s.symbols("bounded_fallback_x bounded_fallback_y")
    program = s.compile_residual_system(
        [x + 2*y + 1, y - 2 + s.Rational(1, 100000000)], (x, y),
    )
    result = program.solve_bounded_least_squares(
        [2.0**-200, 0.0], [(0, None), (None, None)],
    )
    assert result.point[0] == 0.0
    assert result.point[1] == pytest.approx(-1 / 500000000, abs=1e-10)
    assert result.gradient_norm > 0.99
    assert result.projected_gradient_norm <= 1e-10


def test_a_nearby_bound_cannot_authorize_a_strictly_interior_stationary_result():
    program, _ = scalar_program(lambda x: x + 1)
    with pytest.raises(ValueError, match="stationar"):
        program.solve_bounded_least_squares(2.0**-60, [(0, 1)], maxiter=0, gtol=0)
    result = program.solve_bounded_least_squares(0, [(0, 1)], maxiter=0, gtol=0)
    assert result.gradient_norm == 1
    assert result.projected_gradient_norm == 0
    assert result.iterations == 0 and result.evaluations == 2


@pytest.mark.parametrize("endpoints", [(None, None), (-math.inf, math.inf), (-s.oo, s.oo)])
def test_unbounded_endpoint_spellings_retain_actual_binary64_bounds(endpoints):
    program, _ = scalar_program()
    result = program.solve_bounded_least_squares(0, [endpoints])
    assert result.point == pytest.approx((2.0,), abs=1e-10)
    assert result.bounds == ((-math.inf, math.inf),)
    assert result.gradient_norm <= 1e-10


def test_exact_endpoints_are_converted_once_and_reported_as_the_actual_float_box():
    program, _ = scalar_program()
    result = program.solve_bounded_least_squares(
        s.Rational(1, 5), [(s.Rational(1, 10), s.Rational(1, 3))],
    )
    assert result.bounds == ((0.1, 1 / 3),)
    assert result.point == (1 / 3,)
    assert all(type(endpoint) is float for endpoint in result.bounds[0])


def test_mutable_source_layout_and_bounds_cannot_change_a_completed_proposal():
    x = s.Symbol("bounded_same_name", real=True)
    y = s.Symbol("bounded_same_name", positive=True)
    dummy = s.Dummy("bounded_same_name")
    source = s.SparseMatrix([x - 1, y - 2, dummy - 3])
    expected_source = source.as_immutable()
    layout = [(dummy, x), y]
    program = s.compile_residual_system(source, layout)
    encoded = program.to_bytes()
    source[0, 0] = 999
    layout[0] = (x, dummy)
    restored = s.load_residual_system(encoded, residuals=expected_source, args=[(dummy, x), y])
    for owner in (program, restored):
        initial = [0, 0, 2]
        bounds = [[0, 2], [0, 3], [2, 2]]
        result = owner.solve_bounded_least_squares(initial, bounds)
        initial[:] = [999, 999, 999]
        bounds[0][:] = [-999, 999]
        bounds[:] = []
        assert result.point == pytest.approx((2, 1, 2), abs=1e-10)
        assert result.bounds == ((0.0, 2.0), (0.0, 3.0), (2.0, 2.0))
        assert result.variables[0] is dummy
        assert result.variables[1] is x
        assert result.variables[2] is y
        assert result.native_system is owner.native_system
        assert result.cost == pytest.approx(0.5)


def test_candidates_cannot_be_rebound_to_other_sources_or_typed_axes():
    program, x = scalar_program()
    native = program.native_system.solve_bounded_least_squares([0.0], [0.0], [1.0])
    result = s.NumericBoundedLeastSquaresCandidate(native, program.native_system, (x,))
    other = s.compile_residual_system([x - 3], x)
    with pytest.raises(ValueError, match="different residual system"):
        s.NumericBoundedLeastSquaresCandidate(native, other.native_system, (x,))
    other_axis = s.Symbol(x.name, positive=True)
    with pytest.raises(ValueError, match="typed coordinate order"):
        s.NumericBoundedLeastSquaresCandidate(native, program.native_system, (other_axis,))
    with pytest.raises(TypeError):
        s.NumericBoundedLeastSquaresCandidate(object(), program.native_system, (x,))
    with pytest.raises(TypeError):
        type(native)()
    for name, value in [("point", (0.0,)), ("_proposal", native), ("_variables", (other_axis,))]:
        with pytest.raises(AttributeError):
            setattr(result, name, value)
    with pytest.raises(AttributeError):
        del result._native_system
    with pytest.raises(AttributeError):
        result.__init__(native, program.native_system, (x,))
    with pytest.raises(AttributeError):
        native.point = (0.0,)


@pytest.mark.parametrize("kind", ["derivative", "zero_product", "zero_power"])
def test_fixed_coordinates_do_not_erase_original_derivative_or_held_domain_holes(kind):
    x = s.Symbol("bounded_original_domain")
    if kind == "derivative":
        source = s.sqrt(x) - 1
    elif kind == "zero_product":
        source = s.Mul(0, s.log(x), evaluate=False)
    else:
        source = s.Add(s.Pow(s.log(x), 0, evaluate=False), -1, evaluate=False)
    program = s.compile_residual_system([source], x)
    with pytest.raises(ValueError):
        program.solve_bounded_least_squares(0, [(0, 0)])
    with pytest.raises(ValueError):
        program.native_system.solve_bounded_least_squares([0.0], [0.0], [0.0])


def test_fixed_nonroot_and_stationary_maximum_keep_approximate_evidence():
    program, _ = scalar_program()
    fixed = program.solve_bounded_least_squares(1, [(1, 1)], maxiter=0)
    assert fixed.point == (1.0,) and fixed.cost == 0.5
    assert fixed.gradient_norm == 1 and fixed.projected_gradient_norm == 0
    maximum, _ = scalar_program(lambda x: x*x - 1)
    result = maximum.solve_bounded_least_squares(0, [(-1, 1)], maxiter=0)
    assert result.cost == 0.5 and result.projected_gradient_norm == 0
    assert fixed.evidence == result.evidence == "approximate"
    assert not hasattr(fixed, "certificate") and not hasattr(result, "certificate")


@pytest.mark.parametrize("bounds", [None, 1, [], [(0, 1), (0, 1)], [0], [(0,)], [(0, 1, 2)]])
def test_malformed_box_shape_refuses(bounds):
    program, _ = scalar_program()
    with pytest.raises((TypeError, ValueError)):
        program.solve_bounded_least_squares(0, bounds)


@pytest.mark.parametrize("bounds", [
    [(2, 1)], [(math.nan, 1)], [(0, math.nan)], [(math.inf, math.inf)],
    [(-math.inf, -math.inf)], [(s.oo, s.oo)], [(0, -s.oo)],
    [("0", 1)], [(0, "1")], [(1j, 2)],
])
def test_invalid_endpoint_values_refuse(bounds):
    program, _ = scalar_program()
    with pytest.raises((TypeError, ValueError)):
        program.solve_bounded_least_squares(0, bounds)


@pytest.mark.parametrize("initial", [-1, 2, math.nan, math.inf, [], [0, 0], [[0]]])
def test_initial_point_must_be_finite_and_already_feasible(initial):
    program, _ = scalar_program()
    with pytest.raises((TypeError, ValueError)):
        program.solve_bounded_least_squares(initial, [(0, 1)])


def test_closed_numeric_conversion_never_calls_custom_input_or_container_hooks():
    program, _ = scalar_program()

    class Number:
        def __float__(self):
            raise AssertionError("custom number conversion ran")

    class Container(list):
        def __len__(self):
            raise AssertionError("custom container length ran")

        def __iter__(self):
            raise AssertionError("custom container iteration ran")

    for initial, bounds in [
        (Number(), [(0, 1)]), (0, [(Number(), 1)]), (0, [(0, Number())]),
        (Container([0]), [(0, 1)]), (0, Container([(0, 1)])), (0, [Container([0, 1])]),
    ]:
        with pytest.raises((TypeError, ValueError)):
            program.solve_bounded_least_squares(initial, bounds)
    for initial, lower, upper in [
        ([Number()], [0.0], [1.0]), ([0.0], [Number()], [1.0]),
        ([0.0], [0.0], [Number()]), (Container([0.0]), [0.0], [1.0]),
        ([0.0], Container([0.0]), [1.0]), ([0.0], [0.0], Container([1.0])),
    ]:
        with pytest.raises((TypeError, ValueError)):
            program.native_system.solve_bounded_least_squares(initial, lower, upper)


@pytest.mark.parametrize("options", [
    {"gtol": -1}, {"gtol": math.nan}, {"gtol": True},
    {"maxiter": -1}, {"maxiter": 1025}, {"maxiter": True},
    {"max_step_norm": 0}, {"max_step_norm": math.inf}, {"max_step_norm": True},
])
def test_invalid_controls_refuse_without_spending_native_work(options):
    program, _ = scalar_program()
    for native in (False, True):
        budget = s.EvaluationBudget()
        before = (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)
        with pytest.raises((TypeError, ValueError)):
            if native:
                program.native_system.solve_bounded_least_squares(
                    [0.0], [0.0], [1.0], budget=budget, **options,
                )
            else:
                program.solve_bounded_least_squares(0, [(0, 1)], budget=budget, **options)
        assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


@pytest.mark.parametrize("limit,remaining", [
    ("max_steps", "remaining_steps"),
    ("max_memory_bytes", "remaining_memory_bytes"),
    ("max_allocations", "remaining_allocations"),
])
def test_all_rejected_trials_and_publication_use_one_cumulative_budget(limit, remaining):
    program, _ = scalar_program(lambda x: s.log(x))
    measured = s.EvaluationBudget()
    before = getattr(measured, remaining)
    result = program.solve_bounded_least_squares(10, [(0, 20)], budget=measured)
    spent = before - getattr(measured, remaining)
    assert spent > 0
    assert result.evaluations > result.iterations + 2
    exact = s.EvaluationBudget(**{limit: spent})
    assert program.solve_bounded_least_squares(10, [(0, 20)], budget=exact).point == result.point
    assert getattr(exact, remaining) == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_bounded_least_squares(10, [(0, 20)], budget=exact)
    short = s.EvaluationBudget(**{limit: spent - 1})
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_bounded_least_squares(10, [(0, 20)], budget=short)


def test_cancelled_calls_publish_no_candidate_at_either_entrypoint():
    program, _ = scalar_program()
    for native in (False, True):
        budget = s.EvaluationBudget()
        budget.cancel()
        with pytest.raises(s.EvaluationCancelled):
            if native:
                program.native_system.solve_bounded_least_squares([0.0], [0.0], [1.0], budget=budget)
            else:
                program.solve_bounded_least_squares(0, [(0, 1)], budget=budget)


def test_cancellation_after_real_native_work_still_blocks_shell_publication(monkeypatch):
    program, _ = scalar_program()
    budget = s.EvaluationBudget()
    module = importlib.import_module("sympy.utilities._bounded_least_squares")
    original = module.NumericBoundedLeastSquaresCandidate
    constructed = []

    def cancel_after_construction(*args):
        # The actual native solver and actual candidate constructor both run.
        # Only the cancellation schedule is injected at the publication edge.
        candidate = original(*args)
        constructed.append(candidate)
        budget.cancel()
        return candidate

    monkeypatch.setattr(module, "NumericBoundedLeastSquaresCandidate", cancel_after_construction)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_bounded_least_squares(0, [(0, 1)], budget=budget)
    assert len(constructed) == 1 and constructed[0].point == (1.0,)


def test_sparse_bounded_fitting_reaches_256_coordinates_without_dropping_rows():
    variables = s.symbols("bounded_wide_axis_0:256")
    targets = tuple((index % 9) - 4 for index in range(len(variables)))
    residuals = [
        weight*(variable - target)
        for weight in (1, 2)
        for variable, target in zip(variables, targets)
    ]
    program = s.compile_residual_system(residuals, variables)
    result = program.solve_bounded_least_squares([0]*len(variables), [(0, 2)]*len(variables))
    expected = tuple(min(2, max(0, target)) for target in targets)
    assert result.point == pytest.approx(expected, abs=1e-9)
    assert len(result.residuals) == 512
    assert result.cost == pytest.approx(
        sum(5*(value - target)**2 / 2 for value, target in zip(expected, targets)),
    )
    assert result.gradient_norm > 0
    assert result.projected_gradient_norm <= 1e-10
