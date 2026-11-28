"""Real-extension checks for fitting through an exact source-bound objective."""

import pytest
import sympy as s


def test_inconsistent_observations_have_a_certified_positive_minimum():
    x = s.Symbol("fit_observation")
    residuals = s.compile_residual_system([x - 1, x - 3], x)
    objective = residuals.least_squares_objective()
    value, gradient, hessian = objective.evaluate(2)
    assert value == 1
    assert gradient == [0]
    assert hessian == [[2]]
    receipt = objective.minimize_certified(0)
    assert receipt.kind == "strict_minimum"
    lo, hi = receipt.stationary_enclosure[0]
    assert lo <= 2 <= hi
    lo, hi = receipt.objective_enclosure
    assert lo <= 1 <= hi
    # The accepted claim concerns the fitting objective, not roots of both rows.
    assert residuals.residuals(2) == [1, -1]


@pytest.mark.parametrize("point,value,gradient,hessian", [(0, 0.5, 0, -2), (2, 4.5, 12, 22)])
def test_full_nonlinear_curvature_keeps_the_residual_second_derivative(point, value, gradient, hessian):
    x = s.Symbol("fit_nonlinear")
    objective = s.compile_residual_system([x**2 - 1], x).least_squares_objective()
    actual, first, second = objective.evaluate(point)
    assert actual == value
    assert first == [gradient]
    assert second == [[hessian]]


def test_a_stationary_squared_error_maximum_is_not_promoted_to_a_minimum():
    x = s.Symbol("fit_stationary_maximum")
    objective = s.compile_residual_system([x**2 - 1], x).least_squares_objective()
    receipt = objective.certify_stationary_point([(-s.Rational(1, 10), s.Rational(1, 10))])
    assert receipt.kind == "strict_maximum"
    with pytest.raises(s.MinimizationSearchError):
        objective.minimize_certified(0, maxiter=0)


def test_nested_layout_mixed_derivatives_and_hessian_products_keep_coordinate_order():
    x, y = s.symbols("fit_mixed_x fit_mixed_y")
    objective = s.compile_residual_system([x*y - 1, x + y - 3], ((y,), (x,))).least_squares_objective()
    assert objective.variables[0] is y and objective.variables[1] is x
    value, gradient, hessian = objective.evaluate([1], [2])
    assert value == 0.5
    assert gradient == [2, 1]
    assert hessian == [[5, 4], [4, 2]]
    assert objective.hessian_vector_product([2, -1], [1], [2]) == [6, 6]


def test_duplicate_constant_and_zero_rows_retain_their_least_squares_weights():
    x = s.Symbol("fit_weights")
    objective = s.compile_residual_system([x - 1, x - 1, s.Integer(3), s.Integer(0)], x).least_squares_objective()
    value, gradient, hessian = objective.evaluate(2)
    assert value == 5.5
    assert gradient == [2]
    assert hessian == [[2]]


def test_composition_uses_the_frozen_rows_after_the_input_container_changes():
    x = s.Symbol("fit_snapshot")
    rows = [x - 1, x - 3]
    program = s.compile_residual_system(rows, x)
    rows[0] = s.log(x)
    rows.append(x**8)
    objective = program.least_squares_objective()
    assert objective.evaluate(0) == (5.0, [-4.0], [[2.0]])


def test_same_name_symbols_and_dummy_owners_are_not_reconstructed_by_name():
    plain = s.Symbol("fit_identity")
    positive = s.Symbol("fit_identity", positive=True)
    left, right = s.Dummy("fit_identity"), s.Dummy("fit_identity")
    variables = (plain, positive, left, right)
    program = s.compile_residual_system(
        [plain - 1, positive - 2, left - 3, right - 4],
        ((plain, positive), (left, right)),
    )
    objective = program.least_squares_objective()
    assert all(actual is original for actual, original in zip(objective.variables, variables))
    value, gradient, hessian = objective.evaluate([1, 2], [3, 4])
    assert value == 0 and gradient == [0, 0, 0, 0]
    assert hessian == [[int(i == j) for j in range(4)] for i in range(4)]
    with pytest.raises(ValueError):
        objective.enclose([(0, 2), (-3, -1), (2, 4), (3, 5)])


def test_captured_assumptions_survive_mutation_of_the_external_symbol():
    x = s.Symbol("fit_saved_assumptions", positive=True)
    facts = dict(x._assumptions)
    program = s.compile_residual_system([x - 1], x)
    try:
        # The certification domain belongs to the original captured declaration.
        x._assumptions.clear()
        x._assumptions["negative"] = True
        objective = program.least_squares_objective()
        assert objective.variables[0] is x
        assert objective.enclose([(1, 1)]).value == (0, 0)
        with pytest.raises(ValueError, match="positive"):
            objective.enclose([(-2, -1)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_inexact_source_stays_approximate_only_after_composition():
    x = s.Symbol("fit_float")
    objective = s.compile_residual_system([x - s.Float("0.5")], x).least_squares_objective()
    assert objective.evaluate(0.5)[0] == 0
    with pytest.raises(ValueError):
        objective.enclose([(0, 1)])
    with pytest.raises(ValueError):
        objective.minimize_certified(0.5)


@pytest.mark.parametrize("kind", ["zero_product", "zero_power"])
def test_original_held_poles_are_checked_even_when_a_row_simplifies_to_zero(kind):
    x = s.Symbol("fit_hidden_domain")
    pole = s.Pow(x, -1, evaluate=False)
    hidden = (s.Mul(0, pole, evaluate=False) if kind == "zero_product"
              else s.Add(s.Pow(pole, 0, evaluate=False), -1, evaluate=False))
    objective = s.compile_residual_system([x, hidden], x).least_squares_objective()
    assert objective.evaluate(1)[0] == 0.5
    with pytest.raises(ValueError):
        objective.evaluate(0)
    with pytest.raises(ValueError):
        objective.enclose([(-1, 1)])


def test_piecewise_inactive_domains_remain_lazy_in_both_derivative_orders():
    x = s.Symbol("fit_lazy_piecewise")
    row = s.Piecewise((s.log(x), x > 0), (x - 1, True), evaluate=False)
    objective = s.compile_residual_system([row], x).least_squares_objective()
    assert objective.evaluate(-1) == (2.0, [-2.0], [[1.0]])
    enclosure = objective.enclose([(-2, -1)])
    assert enclosure.value[0] <= 2 <= enclosure.value[1]


def test_derived_objective_supports_the_existing_checked_transport():
    from sympy.core import _wrap

    x = s.Symbol("fit_transport")
    program = s.compile_residual_system([x - 1, x - 3], x)
    objective = program.least_squares_objective()
    source = _wrap(objective.native_objective.source)
    restored = s.load_objective(objective.to_bytes(), objective=source, args=x)
    assert restored.evaluate(2) == objective.evaluate(2)
    assert restored.certify_stationary_point([(1, 3)]).kind == "strict_minimum"
    with pytest.raises(ValueError):
        s.load_objective(objective.to_bytes(), objective=(x - 1)**2, args=x)


def test_combined_objective_size_is_bounded_across_all_rows():
    x = s.Symbol("fit_total_bound")
    program = s.compile_residual_system([x] * 85, x)
    with pytest.raises(ValueError):
        program.least_squares_objective()


def test_composition_evaluation_and_certification_share_the_supplied_ledger():
    x = s.Symbol("fit_cumulative")
    program = s.compile_residual_system([x - 1, x - 3], x)
    budget = s.EvaluationBudget(max_steps=100_000_000)
    before = budget.remaining_steps
    objective = program.least_squares_objective(budget=budget)
    after_compile = budget.remaining_steps
    assert after_compile < before
    objective.evaluate(2, budget=budget)
    after_evaluation = budget.remaining_steps
    assert after_evaluation < after_compile
    objective.certify_stationary_point([(1, 3)], budget=budget)
    assert budget.remaining_steps < after_evaluation
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.least_squares_objective(budget=s.EvaluationBudget(max_steps=0))
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.least_squares_objective(budget=cancelled)


def test_fractional_observation_weights_change_the_certified_fit():
    from fractions import Fraction

    x = s.Symbol("weighted_fit")
    observations = s.compile_residual_system([x - 1, x - 4], x)
    weights = [Fraction(1, 2), s.Rational(1, 4)]
    objective = observations.least_squares_objective(weights=weights)
    weights[:] = [0, 0]
    assert objective.evaluate(2) == (0.75, [0.0], [[0.75]])
    receipt = objective.minimize_certified(0)
    assert receipt.kind == "strict_minimum"
    lower, upper = receipt.stationary_enclosure[0]
    assert lower <= 2 <= upper


def test_zero_weight_keeps_singular_observations_in_the_source():
    x = s.Symbol("weighted_domain")
    observations = s.compile_residual_system([x, 1/x], x)
    objective = observations.least_squares_objective(weights=(2, 0))
    assert objective.evaluate(2) == (4.0, [4.0], [[2.0]])
    with pytest.raises(ValueError):
        objective.evaluate(0)


@pytest.mark.parametrize("weights,error", [
    ([1], TypeError), ((1, -1), ValueError), ((True, 0), TypeError),
    ((0.5, 1), TypeError), ((s.Symbol("w"), 0), TypeError),
])
def test_weighted_fit_refuses_malformed_weights(weights, error):
    x = s.Symbol("weighted_input")
    observations = s.compile_residual_system([x, x-1], x)
    with pytest.raises(error):
        observations.least_squares_objective(weights=weights)
