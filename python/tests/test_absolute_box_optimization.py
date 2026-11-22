"""Global objective-value proofs for continuous absolute-value corners."""

from fractions import Fraction

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding


def ample_budget(**limits):
    return s.EvaluationBudget(**{
        "max_steps": 1 << 45,
        "max_memory_bytes": 1 << 40,
        "max_allocations": 1 << 30,
        **limits,
    })


@pytest.mark.parametrize("kind, sign", (("minimum", 1), ("maximum", -1)))
def test_exact_corner_optimum_has_a_global_value_receipt_without_a_gradient(kind, sign):
    x = s.Symbol("absolute_global_corner")
    source = sign*s.Abs(x)
    for result in (
        s.optimize_box_certified(source, x, [(-1, 1)], kind=kind, max_leaves=1),
        s.compile_objective(source, x).optimize_box_certified(
            [(-1, 1)], kind=kind, max_leaves=1,
        ),
    ):
        assert type(result) is s.BoxOptimizationReceipt
        assert result.objective == source
        assert result.kind == kind
        assert result.domain == ((-1, 1),)
        assert result.point == (0,)
        assert result.value == result.point_value == (0, 0)
        assert result.gap == 0 and result.leaf_count == 1
        assert "global" in result.claim and "gap" in result.claim
        assert not hasattr(result, "gradient")
        assert not hasattr(result, "hessian")
        assert not hasattr(result, "uniqueness_proved")


def test_multivariate_absolute_loss_preserves_typed_coordinates_and_assumptions():
    plain = s.Symbol("absolute_global_typed")
    positive = s.Symbol("absolute_global_typed", positive=True)
    dummy = s.Dummy("absolute_global_typed")
    variables = (plain, positive, dummy)
    source = s.Abs(plain) + 2*s.Abs(positive-2) + 3*s.Abs(dummy)
    result = s.optimize_box_certified(
        source, variables, [(-1, 1), (1, 3), (-2, 2)], max_leaves=1,
    )
    assert result.point == (0, 2, 0)
    assert result.value == result.point_value == (0, 0)
    assert result.leaf_count == 1
    for actual, variable in zip(result.variables, variables):
        expected = _symbol_binding(variable)
        assert actual.name == expected.name
        assert actual.plain == expected.plain
        if not actual.plain:
            assert actual.identity_hex == expected.identity_hex
    with pytest.raises(ValueError, match="positive"):
        s.optimize_box_certified(source, variables, [(-1, 1), (-1, 3), (-2, 2)])


def test_shifted_nondyadic_corner_subdivides_to_an_honest_value_gap():
    x = s.Symbol("absolute_global_shifted")
    shift = s.Rational(1, 3)
    source = s.Abs(x-shift)
    tolerance = Fraction(1, 1024)
    result = s.optimize_box_certified(
        source, x, [(-1, 1)], tolerance=tolerance, max_leaves=128,
    )
    actual = abs(result.point[0]-shift)
    assert result.point[0] != shift
    assert result.value[0] <= 0 <= result.value[1]
    assert result.point_value == (actual, actual)
    assert 0 < result.gap == result.value[1]-result.value[0] <= s.Rational(1, 1024)
    assert result.absolute_tolerance == s.Rational(1, 1024)
    assert result.leaf_count > 1
    with pytest.raises(ValueError, match="gap|tolerance"):
        s.optimize_box_certified(source, x, [(-1, 1)], tolerance=tolerance, max_leaves=1)


@pytest.mark.parametrize("form", ("nested", "exponential", "reciprocal"))
def test_smooth_outer_functions_compose_with_the_inner_absolute_corner(form):
    x = s.Symbol("absolute_global_composition")
    source, kind, value = {
        "nested": (s.Abs(s.Abs(x), evaluate=False), "minimum", 0),
        "exponential": (s.exp(s.Abs(x)), "minimum", 1),
        "reciprocal": (1/(1+s.Abs(x)), "maximum", 1),
    }[form]
    result = s.optimize_box_certified(source, x, [(-1, 1)], kind=kind, max_leaves=1)
    assert result.point == (0,)
    assert result.value == result.point_value == (value, value)
    assert result.gap == 0 and result.leaf_count == 1


def test_boundary_maximum_uses_smooth_cells_alongside_a_nonsmooth_cell():
    x = s.Symbol("absolute_global_boundary")
    result = s.optimize_box_certified(s.Abs(x), x, [(-1, 2)], kind="maximum")
    assert result.point == (2,)
    assert result.value == result.point_value == (2, 2)
    assert result.gap == 0


@pytest.mark.parametrize("form, expected", (("zero_product", 0), ("zero_power", 1)))
def test_defined_held_absolute_values_are_admitted_at_the_corner(form, expected):
    x = s.Symbol("absolute_global_held")
    absolute = s.Abs(x, evaluate=False)
    source = (s.Mul(0, absolute, evaluate=False) if form == "zero_product"
              else s.Pow(s.Add(1, absolute, evaluate=False), 0, evaluate=False))
    result = s.optimize_box_certified(source, x, [(-1, 1)], max_leaves=1)
    assert result.objective == source
    assert result.value == result.point_value == (expected, expected)
    assert result.gap == 0


@pytest.mark.parametrize("form", ("reciprocal", "held_reciprocal", "log", "zero_over_x", "zero_power"))
def test_original_child_poles_block_global_acceptance_even_behind_zero_factors(form):
    x = s.Symbol("absolute_global_domain")
    inverse = s.Pow(x, -1, evaluate=False)
    source = {
        "reciprocal": s.Abs(inverse, evaluate=False),
        "held_reciprocal": s.Add(s.Abs(x), s.Mul(0, inverse, evaluate=False), evaluate=False),
        "log": s.Abs(s.log(x), evaluate=False),
        "zero_over_x": s.Abs(s.Mul(0, inverse, evaluate=False), evaluate=False),
        "zero_power": s.Pow(s.Abs(x, evaluate=False), 0, evaluate=False),
    }[form]
    with pytest.raises(ValueError):
        s.optimize_box_certified(source, x, [(-1, 1)], max_leaves=8)


def test_global_corner_support_does_not_weaken_derivative_or_stationarity_claims():
    x = s.Symbol("absolute_global_strict")
    source = s.Abs(x)
    assert s.optimize_box_certified(source, x, [(-1, 1)]).value == (0, 0)
    for operation in (
        lambda: s.enclose_residual_system(source, x, [(-1, 1)]),
        lambda: s.enclose_objective(source, x, [(-1, 1)]),
        lambda: s.certify_root_box(source, x, [(-1, 1)]),
        lambda: s.certify_stationary_point(source, x, [(-1, 1)]),
        lambda: s.certify_box_extremum(source, x, [(-1, 1)], [0]),
    ):
        with pytest.raises(ValueError):
            operation()


def test_direct_global_source_path_does_not_need_numeric_derivative_programs(monkeypatch):
    x = s.Symbol("absolute_global_source")

    def forbidden(*args, **kwargs):
        raise AssertionError("global value checking must not compile derivative programs")

    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    result = s.optimize_box_certified(s.Abs(x), x, [(-1, 1)])
    assert result.value == (0, 0)


def test_stable_piecewise_guards_keep_inactive_domains_lazy_and_seams_conservative():
    x = s.Symbol("absolute_global_piecewise")
    pole = s.Pow(s.Add(x, -x, evaluate=False), -1, evaluate=False)
    stable = s.Piecewise((s.Abs(x), x < 2), (pole, True), evaluate=False)
    assert s.optimize_box_certified(stable, x, [(-1, 1)]).value == (0, 0)
    seam = s.Piecewise((s.Abs(x), x > 0), (-x, True), evaluate=False)
    with pytest.raises(ValueError):
        s.optimize_box_certified(seam, x, [(-1, 1)], max_leaves=8)


def test_nonsmooth_optimization_retains_cumulative_budget_and_cancellation():
    x = s.Symbol("absolute_global_budget")
    source = s.Abs(x)
    budget = ample_budget()
    before = budget.remaining_steps
    first = s.optimize_box_certified(source, x, [(-1, 1)], budget=budget)
    after = budget.remaining_steps
    assert first.value == (0, 0) and after < before
    assert s.optimize_box_certified(source, x, [(-1, 1)], budget=budget).value == (0, 0)
    assert budget.remaining_steps < after
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.optimize_box_certified(source, x, [(-1, 1)], budget=ample_budget(max_steps=0))
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        s.optimize_box_certified(source, x, [(-1, 1)], budget=cancelled)
