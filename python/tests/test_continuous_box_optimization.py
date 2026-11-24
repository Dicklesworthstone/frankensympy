"""Global value bounds at continuous endpoints without derivative claims."""

from fractions import Fraction

import pytest
import sympy as s

from sympy.core import _native


def ample_budget(**changes):
    return s.EvaluationBudget(**{
        "max_steps": 1 << 45,
        "max_memory_bytes": 1 << 40,
        "max_allocations": 1 << 30,
        **changes,
    })


def objective(family, x):
    return {
        "sqrt": lambda: s.sqrt(x),
        "one_third": lambda: s.Pow(x, s.Rational(1, 3), evaluate=False),
        "two_thirds": lambda: s.Pow(x, s.Rational(2, 3), evaluate=False),
        "acos": lambda: s.acos(x),
        "acosh": lambda: s.acosh(x),
        "asin_square": lambda: s.Pow(s.asin(x), 2, evaluate=False),
    }[family]()


@pytest.mark.parametrize("family,box,kind,expected", [
    ("sqrt", (0, 4), "minimum", 0),
    ("sqrt", (0, 4), "maximum", 2),
    ("one_third", (0, 8), "minimum", 0),
    ("two_thirds", (0, 27), "maximum", 9),
    ("acos", (0, 1), "minimum", 0),
    ("acosh", (1, 2), "minimum", 0),
    ("asin_square", (-1, 1), "minimum", 0),
])
def test_direct_source_search_proves_known_continuous_extrema(family, box, kind, expected):
    x = s.Symbol("continuous_global_endpoint")
    source = objective(family, x)
    result = s.optimize_box_certified(
        source, x, [box], kind=kind, tolerance=Fraction(1, 16),
        precision=16, max_leaves=128, budget=ample_budget(),
    )
    assert type(result) is s.BoxOptimizationReceipt
    assert result.objective == source
    assert result.domain == (box,) and result.kind == kind
    assert result.value[0] <= expected <= result.value[1]
    assert 0 <= result.gap <= s.Rational(1, 16)
    assert result.absolute_tolerance == s.Rational(1, 16)
    assert box[0] <= result.point[0] <= box[1]
    assert not hasattr(result, "gradient")
    assert not hasattr(result, "hessian")
    assert not hasattr(result, "stationary_enclosure")


def test_interior_nonlipschitz_corner_needs_no_symbolic_derivative_compiler(monkeypatch):
    x = s.Symbol("continuous_global_corner")

    def forbidden(*args, **kwargs):
        raise AssertionError("continuous value bounds must not compile derivative programs")

    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    source = s.sqrt(s.Abs(x))
    result = s.optimize_box_certified(
        source, x, [(-1, 1)], precision=16, max_leaves=1, budget=ample_budget(),
    )
    assert result.objective == source
    assert result.point == (0,)
    assert result.value == result.point_value == (0, 0)
    assert result.gap == 0 and result.leaf_count == 1


@pytest.mark.parametrize("form,expected", [("zero_product", 0), ("zero_power", 1)])
def test_defined_held_endpoint_values_survive_original_source_validation(form, expected):
    x = s.Symbol("continuous_global_held")
    root = s.Pow(x, s.Rational(1, 2), evaluate=False)
    source = (
        s.Mul(0, root, evaluate=False) if form == "zero_product"
        else s.Pow(s.Add(1, root, evaluate=False), 0, evaluate=False)
    )
    result = s.optimize_box_certified(
        source, x, [(0, 1)], precision=16, max_leaves=1, budget=ample_budget(),
    )
    assert result.objective == source
    assert result.value == result.point_value == (expected, expected)
    assert result.gap == 0


@pytest.mark.parametrize("family,box", [
    ("sqrt", (0, 1)),
    ("one_third", (0, 1)),
    ("acos", (0, 1)),
    ("acosh", (1, 2)),
])
def test_value_admission_never_grants_endpoint_derivatives(family, box):
    x = s.Symbol("continuous_global_strict")
    source = objective(family, x)
    s.optimize_box_certified(
        source, x, [box], tolerance=Fraction(1, 16),
        precision=16, max_leaves=128, budget=ample_budget(),
    )
    for operation in (
        lambda: s.enclose_residual_system(source, x, [box], precision=16),
        lambda: s.enclose_objective(source, x, [box], precision=16),
        lambda: s.certify_root_box(source, x, [box], precision=16),
        lambda: s.certify_stationary_point(source, x, [box], precision=16),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("form", ["held_pole", "nonreal", "negative_power", "piecewise_seam"])
def test_true_domain_failures_cannot_hide_behind_continuous_retry(form):
    x = s.Symbol("continuous_global_domain")
    source = {
        "held_pole": lambda: s.Add(
            s.sqrt(x), s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False),
            evaluate=False,
        ),
        "nonreal": lambda: s.sqrt(x-1),
        "negative_power": lambda: s.Pow(x, s.Rational(-1, 2), evaluate=False),
        "piecewise_seam": lambda: s.Piecewise(
            (s.sqrt(x), x < s.Rational(1, 2)), (10, True), evaluate=False,
        ),
    }[form]()
    with pytest.raises(ValueError):
        s.optimize_box_certified(
            source, x, [(0, 1)], tolerance=Fraction(1, 16),
            precision=16, max_leaves=8, budget=ample_budget(),
        )


def test_range_only_checkpoint_resumes_against_its_exact_original_source():
    x = s.Symbol("continuous_global_resume", real=True)
    source = s.sqrt(s.Abs(x-s.Rational(1, 3)))
    options = dict(tolerance=Fraction(1, 32), precision=16, max_leaves=128)
    search = s.begin_box_optimization(
        source, x, [(0, 1)], budget=ample_budget(), **options,
    )
    assert not search.progress.ready_for_verification
    search.advance(2)
    data = search.checkpoint()
    restored = s.restore_box_optimization(
        data, objective=source, args=x, box=[(0, 1)],
        budget=ample_budget(), **options,
    )
    assert restored.objective == source
    assert restored.progress.evaluation_count >= search.progress.evaluation_count
    for _ in range(128):
        if restored.progress.ready_for_verification:
            break
        restored.advance(3)
    assert restored.progress.ready_for_verification
    result = restored.finish()
    blocking = s.optimize_box_certified(
        source, x, [(0, 1)], budget=ample_budget(), **options,
    )
    assert result.point == blocking.point
    assert result.value == blocking.value
    assert result.gap == blocking.gap
    assert result.value[0] <= 0 <= result.value[1]
    assert 0 < result.gap <= s.Rational(1, 32)
    with pytest.raises(ValueError):
        s.restore_box_optimization(
            data, objective=s.sqrt(s.Abs(x-s.Rational(1, 4))),
            args=x, box=[(0, 1)], budget=ample_budget(), **options,
        )


def test_continuous_search_keeps_cancellation_and_exhaustion_terminal():
    x = s.Symbol("continuous_global_budget")
    source = s.sqrt(s.Abs(x-s.Rational(1, 3)))
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.optimize_box_certified(
            source, x, [(0, 1)], budget=ample_budget(max_steps=0),
        )
    budget = ample_budget()
    search = s.begin_box_optimization(
        source, x, [(0, 1)], tolerance=Fraction(1, 32),
        precision=16, max_leaves=128, budget=budget,
    )
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        search.advance(1)
