"""Principal inverse-circular claims use the original exact source and box."""

import pytest
import sympy as s

from sympy.core import _native


ENCLOSERS = (s.enclose_residual_system, s.enclose_objective)


def contains(interval, value):
    assert interval[0] <= value <= interval[1]


def tight(interval, value):
    contains(interval, value)
    assert interval[1] - interval[0] <= s.Rational(1, 10**24)


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


@pytest.mark.parametrize("name,sign", [("asin", 1), ("acos", -1)])
def test_exact_source_rules_include_both_nonlinear_chain_terms(name, sign, monkeypatch):
    x = s.Symbol("inverse_chain_x")
    source = getattr(s, name)(s.Rational(3, 5) + 2*x + 3*x*x)

    def forbidden(*args, **kwargs):
        raise AssertionError("original-source certification must not compile derivatives")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    assert residual.residuals == (source,) and objective.objective == source
    tight(residual.jacobian[0][0], sign*s.Rational(5, 2))
    tight(objective.gradient[0], sign*s.Rational(5, 2))
    # f''(u)*(u')^2 + f'(u)*u'': (75/64)*4 + (5/4)*6.
    tight(objective.hessian[0][0], sign*s.Rational(195, 16))
    assert residual.residual_values[0] == objective.value


def test_mixed_partials_and_arctangent_remain_composable():
    x, y = s.symbols("inverse_mixed_x inverse_mixed_y")
    source = s.asin(s.Rational(3, 5) + x*y) + s.atan(1 + x)
    result = s.enclose_objective(source, (x, y), [(0, 0), (0, 0)])
    tight(result.gradient[0], s.Rational(1, 2))
    tight(result.gradient[1], 0)
    tight(result.hessian[0][0], s.Rational(-1, 2))
    tight(result.hessian[0][1], s.Rational(5, 4))
    tight(result.hessian[1][0], s.Rational(5, 4))
    tight(result.hessian[1][1], 0)


@pytest.mark.parametrize("name", ["asin", "acos", "atan"])
def test_compiled_and_direct_enclosures_use_the_same_original_source(name):
    x = s.Symbol("inverse_shared_source")
    source = getattr(s, name)(x)
    box = [(s.Rational(1, 4), s.Rational(1, 2))]
    direct = s.enclose_residual_system(source, x, box)
    compiled = s.compile_residual_system(source, x).enclose(box)
    assert direct.residual_values == compiled.residual_values
    assert direct.jacobian == compiled.jacobian
    direct_objective = s.enclose_objective(source, x, box)
    compiled_objective = s.compile_objective(source, x).enclose(box)
    assert direct_objective.value == compiled_objective.value
    assert direct_objective.gradient == compiled_objective.gradient
    assert direct_objective.hessian == compiled_objective.hessian


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("name", ["asin", "acos"])
@pytest.mark.parametrize("bounds", [
    (-1, -1), (1, 1), (-1, 0), (0, 1), (-1, 1),
    (1 + s.Rational(1, 10**80), 1 + s.Rational(1, 10**80)),
    (-1 - s.Rational(1, 10**80), -1 - s.Rational(1, 10**80)),
])
def test_c1_and_c2_require_the_entire_argument_box_strictly_inside_domain(operation, name, bounds):
    x = s.Symbol("inverse_endpoint_x")
    with pytest.raises(ValueError):
        operation(getattr(s, name)(x), x, [bounds])


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("name", ["asin", "acos"])
@pytest.mark.parametrize("held", ["zero_product", "zero_power"])
def test_held_zero_operations_cannot_erase_inverse_derivative_domains(operation, name, held):
    x = s.Symbol("inverse_held_x")
    application = getattr(s, name)(x)
    source = (s.Mul(0, application, evaluate=False) if held == "zero_product"
              else s.Pow(application, 0, evaluate=False))
    for point in (-1, 1, 2):
        with pytest.raises(ValueError):
            operation(source, x, [(point, point)])
    # Both inverse values are nonzero here. At x=0, held asin(x)**0
    # instead has the independently forbidden original-source form 0**0.
    interior = s.Rational(1, 2)
    result = operation(source, x, [(interior, interior)])
    expected = 0 if held == "zero_product" else 1
    value = result.residual_values[0] if operation is s.enclose_residual_system else result.value
    assert value == (expected, expected)


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("name", ["asin", "atan"])
def test_held_inverse_zero_powers_preserve_indeterminate_zero_base(operation, name):
    x = s.Symbol("inverse_zero_base_x")
    source = s.Pow(getattr(s, name)(x), 0, evaluate=False)
    with pytest.raises(ValueError, match="zero exponent"):
        operation(source, x, [(0, 0)])


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_piecewise_checks_only_the_stably_selected_inverse_value(operation):
    x = s.Symbol("inverse_piecewise_x")
    source = s.Piecewise((s.asin(x), x > 0), (s.asin(2*x), True), evaluate=False)
    box = [(s.Rational(3, 5), s.Rational(3, 4))]
    selected = operation(source, x, box)
    reference = operation(s.asin(x), x, box)
    if operation is s.enclose_residual_system:
        assert selected.residual_values == reference.residual_values
        assert selected.jacobian == reference.jacobian
    else:
        assert selected.value == reference.value
        assert selected.gradient == reference.gradient
        assert selected.hessian == reference.hessian
    # The second branch is outside [-1,1] here and is now selected.
    with pytest.raises(ValueError):
        operation(source, x, [(s.Rational(-3, 4), s.Rational(-3, 5))])
    with pytest.raises(ValueError):
        operation(source, x, [(s.Rational(-1, 4), s.Rational(1, 4))])


@pytest.mark.parametrize("name", ["asin", "acos", "atan"])
def test_root_receipts_bind_the_full_inverse_source_and_reject_false_boxes(name):
    x = s.Symbol("inverse_root_x")
    source = getattr(s, name)(x)
    if name == "acos":
        source -= s.pi/2
    box = [(s.Rational(-1, 4), s.Rational(1, 4))]
    receipt = s.certify_root_box(source, x, box)
    assert receipt.residuals == (source,)
    assert receipt.domain == tuple(box)
    contains(receipt.root_enclosure[0], 0)
    assert receipt.contraction_bound < 1
    with pytest.raises(ValueError):
        s.certify_root_box(source, x, [(s.Rational(1, 4), s.Rational(1, 2))])
    with pytest.raises(ValueError):
        s.certify_root_box(source, x, box, preconditioner=[[0]])


def test_arcsine_squared_has_an_independently_classified_local_minimum():
    x = s.Symbol("inverse_minimum_x")
    source = s.asin(x)**2
    box = [(s.Rational(-1, 20), s.Rational(1, 20))]
    receipt = s.certify_stationary_point(source, x, box)
    assert receipt.objective == source and receipt.domain == tuple(box)
    assert receipt.kind == "strict_minimum"
    contains(receipt.stationary_enclosure[0], 0)
    assert receipt.contraction_bound < 1
    with pytest.raises(ValueError):
        s.certify_stationary_point(source, x, [(-1, 1)])


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("name", ["asin", "acos", "atan"])
def test_undefined_intrinsic_spelling_does_not_gain_certificate_authority(operation, name):
    x = s.Symbol("inverse_opaque_x")
    with pytest.raises(NotImplementedError):
        operation(s.Function(name)(x), x, [(0, 0)])


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_inverse_enclosures_share_all_budget_dimensions_across_calls(operation):
    x = s.Symbol("inverse_budget_x")
    source = s.asin(x) + s.acos(x)
    budget = ample_budget()
    fields = ("remaining_steps", "remaining_memory_bytes", "remaining_allocations")
    before = tuple(getattr(budget, field) for field in fields)
    first = operation(source, x, [(s.Rational(1, 5), s.Rational(2, 5))], budget=budget)
    after = tuple(getattr(budget, field) for field in fields)
    second = operation(source, x, [(s.Rational(1, 5), s.Rational(2, 5))], budget=budget)
    final = tuple(getattr(budget, field) for field in fields)
    assert all(end < middle < start for start, middle, end in zip(before, after, final))
    assert first.domain == second.domain


@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
@pytest.mark.parametrize("operation", ENCLOSERS)
def test_inverse_publication_refuses_exhaustion_and_precancelled_requests(operation, limit):
    x = s.Symbol("inverse_stopped_x")
    source = s.asin(x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        operation(source, x, [(0, 0)], budget=ample_budget(**{limit: 0}))
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        operation(source, x, [(0, 0)], budget=cancelled)
