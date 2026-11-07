"""Actual-extension regressions for bounded checked-objective minimization."""

import math

import pytest
import sympy as s
from sympy.core import _native
from sympy.utilities import NumericMinimizationCandidate


def test_coupled_quadratic_has_fresh_source_bound_diagnostics():
    x, y = s.symbols("minimize_coupled_x minimize_coupled_y")
    source = 2*(x-1)**2 + (x-1)*(y+2) + 3*(y+2)**2
    program = s.compile_objective(source, (x, y))
    result = program.minimize([7, 9])
    assert type(result) is NumericMinimizationCandidate is s.NumericMinimizationCandidate
    assert result.point == pytest.approx((1, -2), abs=1e-10)
    assert result.objective == source
    assert result.variables[0] is x and result.variables[1] is y
    assert result.value == pytest.approx(0, abs=1e-20)
    assert result.gradient_norm <= 1e-10
    value, gradient, _ = program.evaluate(*result.point)
    assert result.value == value
    assert result.gradient == tuple(gradient)
    assert result.gradient_norm == max(map(abs, gradient))
    assert result.iterations > 0 and result.evaluations >= result.iterations + 2
    assert result.termination == "gradient_tolerance"
    assert result.backend == "frankennumpy" and result.evidence == "approximate"
    assert not hasattr(result, "certificate") and not hasattr(result, "kind")


def test_rosenbrock_uses_the_checked_full_hessian():
    x, y = s.symbols("minimize_rosenbrock_x minimize_rosenbrock_y")
    program = s.compile_objective((1-x)**2 + 100*(y-x*x)**2, (x, y))
    result = program.minimize([-1.2, 1])
    assert result.point == pytest.approx((1, 1), abs=1e-8)
    assert result.value >= 0 and result.value < 1e-16
    assert result.gradient_norm <= 1e-10
    assert result.evaluations >= result.iterations + 2


def test_indefinite_initial_hessian_uses_regularized_descent():
    x = s.Symbol("minimize_nonconvex")
    result = s.compile_objective(x**4-x*x, x).minimize(0.1)
    assert result.point == pytest.approx((math.sqrt(0.5),), abs=1e-8)
    assert result.value == pytest.approx(-0.25, abs=1e-14)
    assert result.regularizations > 0
    assert result.gradient_norm <= 1e-10


def test_domain_crossing_trials_backtrack_and_count_rejected_evaluations():
    x = s.Symbol("minimize_logarithm")
    program = s.compile_objective(x-s.log(x), x)
    # At 10, gradient=.9 and Hessian=.01, so the full Newton step is -90.
    result = program.minimize(10)
    assert result.point == pytest.approx((1,), abs=1e-9)
    assert result.value == pytest.approx(1, abs=1e-14)
    assert result.evaluations > result.iterations + 2
    for initial in (0, -1):
        with pytest.raises(ValueError):
            program.minimize(initial)


@pytest.mark.parametrize("kind", ["maximum", "saddle", "constant"])
def test_initial_stationarity_is_only_an_approximate_candidate(kind):
    x, y = s.symbols("minimize_stationary_x minimize_stationary_y")
    source = {"maximum": -x*x-y*y, "saddle": x*x-y*y, "constant": s.Integer(7)}[kind]
    result = s.compile_objective(source, (x, y)).minimize([0, 0], maxiter=0)
    assert result.point == (0.0, 0.0)
    assert result.gradient == (0.0, 0.0)
    assert result.iterations == 0 and result.evaluations == 2
    assert result.evidence == "approximate"
    assert not hasattr(result, "certificate") and not hasattr(result, "kind")


@pytest.mark.parametrize("hidden_kind", ["zero_product", "zero_power"])
def test_original_held_domain_cannot_disappear_at_a_stationary_point(hidden_kind):
    x = s.Symbol("minimize_held_domain")
    hidden = (s.Mul(0, s.log(x), evaluate=False) if hidden_kind == "zero_product"
              else s.Add(s.Pow(s.log(x), 0, evaluate=False), -1, evaluate=False))
    program = s.compile_objective(s.Add(x*x, hidden, evaluate=False), x)
    for call in (
        lambda: program.minimize(0),
        lambda: program.native_objective.minimize([0.0]),
    ):
        with pytest.raises(ValueError):
            call()


def test_second_derivative_domain_is_checked_even_when_gradient_is_zero():
    x = s.Symbol("minimize_second_derivative_domain")
    program = s.compile_objective(x**s.Rational(3, 2), x)
    with pytest.raises(ValueError):
        program.minimize(0)


def test_same_name_assumed_symbols_dummies_and_nested_layout_keep_coordinate_identity():
    plain, positive = s.Symbol("minimize_typed"), s.Symbol("minimize_typed", positive=True)
    left, right = s.Dummy("minimize_typed"), s.Dummy("minimize_typed")
    variables = (plain, positive, left, right)
    source = (plain-1)**2 + (positive-2)**2 + (left-3)**2 + (right-4)**2
    program = s.compile_objective(source, ((plain, positive), (left, right)))
    result = program.minimize([0, 1, 0, 0])
    assert result.point == pytest.approx((1, 2, 3, 4), abs=1e-10)
    assert all(actual is expected for actual, expected in zip(result.variables, variables))
    assert result.objective == source
    assert program.evaluate(result.point[:2], result.point[2:])[0] == result.value
    with pytest.raises(TypeError):
        program.minimize([[0, 1], [0, 0]])


@pytest.mark.parametrize("field", ["point", "objective", "value", "gradient", "iterations"])
def test_native_and_public_proposals_are_immutable(field):
    x = s.Symbol("minimize_immutable")
    program = s.compile_objective((x-1)**2, x)
    result = program.minimize(0)
    proposal = result._proposal
    assert type(proposal) is _native.NumericMinimizationProposal
    assert type(proposal.point) is tuple and type(proposal.gradient) is tuple
    assert proposal.objective is program.native_objective
    for target in (result, proposal):
        with pytest.raises(AttributeError):
            setattr(target, field, None)
        with pytest.raises(AttributeError):
            delattr(target, field)
    with pytest.raises(AttributeError):
        result.__init__(proposal, program.native_objective, (x,))
    with pytest.raises(TypeError):
        _native.NumericMinimizationProposal()


def test_a_native_proposal_cannot_be_relabelled_with_a_different_objective():
    x = s.Symbol("minimize_source_owner")
    first = s.compile_objective((x-1)**2, x)
    second = s.compile_objective((x-2)**2, x)
    native = first.native_objective.minimize([0.0])
    assert native.objective is first.native_objective
    with pytest.raises(ValueError, match="different objective"):
        NumericMinimizationCandidate(native, second.native_objective, (x,))
    with pytest.raises(TypeError):
        NumericMinimizationCandidate(object(), first.native_objective, (x,))


def test_candidate_axes_must_match_captured_typed_identity_and_order():
    plain = s.Symbol("minimize_axis_identity")
    positive = s.Symbol("minimize_axis_identity", positive=True)
    dummy = s.Dummy("minimize_axis_identity")
    other_dummy = s.Dummy("minimize_axis_identity")
    variables = (plain, positive, dummy)
    program = s.compile_objective(
        (plain-1)**2 + (positive-2)**2 + (dummy-3)**2, variables,
    )
    proposal = program.native_objective.minimize([0.0, 1.0, 0.0])
    accepted = NumericMinimizationCandidate(proposal, program.native_objective, variables)
    assert all(a is b for a, b in zip(accepted.variables, variables))
    for axes in ((positive, plain, dummy), (plain, plain, dummy),
                 (plain, positive, other_dummy)):
        with pytest.raises(ValueError, match="typed coordinate order"):
            NumericMinimizationCandidate(proposal, program.native_objective, axes)
    with pytest.raises(TypeError):
        NumericMinimizationCandidate(proposal, program.native_objective, (plain,))


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_mutated_assumption_dict_does_not_change_captured_native_axes(constructor):
    x = constructor("minimize_axis_snapshot", positive=True)
    facts = dict(x._assumptions)
    program = s.compile_objective((x-1)**2, x)
    try:
        x._assumptions.clear()
        x._assumptions["negative"] = True
        result = program.minimize(2)
        assert result.point == pytest.approx((1,), abs=1e-10)
        assert result.variables[0] is x
        assert result._proposal.objective is program.native_objective
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_candidate_axis_admission_refuses_iterators_and_custom_symbol_hooks():
    x = s.Symbol("minimize_axis_admission")
    program = s.compile_objective((x-1)**2, x)
    proposal = program.native_objective.minimize([0.0])
    effects = []

    class Axes:
        def __iter__(self):
            effects.append("iteration")
            yield x

    class Axis:
        def __str__(self):
            effects.append("printer")
            return str(x)

    for axes in (Axes(), (), (x, x), (Axis(),)):
        with pytest.raises(TypeError):
            NumericMinimizationCandidate(proposal, program.native_objective, axes)
    assert not effects


@pytest.mark.parametrize("options", [
    {"gtol": -1}, {"gtol": float("nan")}, {"gtol": float("inf")},
    {"maxiter": -1}, {"maxiter": 1025},
    {"max_step": 0}, {"max_step": float("inf")},
])
def test_invalid_controls_refuse_before_initial_hooks_and_budget_work(options):
    x = s.Symbol("minimize_controls")
    program = s.compile_objective((x-1)**2, x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("invalid controls must not convert initial values")

    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    for target, initial in ((program, Initial()), (program.native_objective, [Initial()])):
        with pytest.raises(ValueError):
            target.minimize(initial, budget=budget, **options)
    assert not effects and budget.remaining_steps == before


@pytest.mark.parametrize("options", [
    {"gtol": True}, {"maxiter": True}, {"max_step": True},
    {"gtol": "1e-10"}, {"maxiter": "64"}, {"max_step": "1e6"},
])
def test_direct_native_controls_require_exact_admitted_builtin_types(options):
    x = s.Symbol("minimize_native_controls")
    native = s.compile_objective((x-1)**2, x).native_objective
    with pytest.raises(TypeError):
        native.minimize([0.0], **options)


def test_iteration_limit_and_stagnation_never_report_false_success():
    x = s.Symbol("minimize_stop")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(ValueError):
        program.minimize(0, maxiter=0)
    result = program.minimize(1, maxiter=0, gtol=0)
    assert result.iterations == 0 and result.gradient_norm == 0
    with pytest.raises(ValueError):
        program.minimize(
            2, gtol=0, max_step=float.fromhex("0x0.0000000000001p-1022"),
        )


def test_shared_budget_is_cumulative_and_cancellation_stays_typed():
    x = s.Symbol("minimize_budget")
    program = s.compile_objective((x-1)**2, x)
    measured = s.EvaluationBudget()
    before = measured.remaining_steps
    program.minimize(0, budget=measured)
    cost = before-measured.remaining_steps
    assert cost > 0
    exact = s.EvaluationBudget(max_steps=cost)
    program.minimize(0, budget=exact)
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize(0, budget=exact)
    short = s.EvaluationBudget(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize(0, budget=short)
    assert short.remaining_steps < cost-1
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.minimize(0, budget=cancelled)


@pytest.mark.parametrize("limit", ["max_memory_bytes", "max_allocations"])
def test_input_buffer_reservation_is_atomic_before_any_domain_evaluation(limit):
    x = s.Symbol("minimize_buffer_budget")
    program = s.compile_objective(s.log(x), x)
    budget = s.EvaluationBudget(**{limit: 0})
    before = (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize(-1, budget=budget)
    assert before == (
        budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations,
    )


@pytest.mark.parametrize("dimension", [0, 17])
def test_unsupported_dimensions_refuse_at_both_entrypoints(dimension):
    variables = tuple(s.Symbol(f"minimize_shape_{dimension}_{i}") for i in range(dimension))
    source = sum((variable*variable for variable in variables), s.Integer(0))
    program = s.compile_objective(source, variables)
    for target in (program, program.native_objective):
        with pytest.raises(ValueError, match="1..16"):
            target.minimize([0]*dimension)


@pytest.mark.parametrize("initial", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_nonfinite_inputs_and_overflowing_original_arithmetic_refuse(initial):
    x = s.Symbol("minimize_nonfinite")
    program = s.compile_objective(x*x, x)
    with pytest.raises(ValueError):
        program.minimize(initial)


def test_initial_coordinate_admission_never_calls_custom_float_hooks():
    x = s.Symbol("minimize_input_hook")
    program = s.compile_objective((x-1)**2, x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            return 0.0

    for target, initial in ((program, Initial()), (program.native_objective, [Initial()])):
        with pytest.raises(TypeError):
            target.minimize(initial)
    assert not effects
