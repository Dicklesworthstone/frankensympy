"""Automatic nonconvex box optimization with independently certified gaps."""

from fractions import Fraction
import importlib

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def rational(value):
    return Fraction(value.p, value.q)


def assert_gap(receipt, tolerance, known_value):
    lower, upper = map(rational, receipt.value)
    point_lower, point_upper = map(rational, receipt.point_value)
    gap = rational(receipt.gap)
    assert lower <= known_value <= upper
    assert 0 <= gap == upper - lower <= tolerance
    assert rational(receipt.absolute_tolerance) == tolerance
    if receipt.kind == "minimum":
        assert upper == point_upper
    else:
        assert lower == point_lower
    assert 1 <= receipt.leaf_count <= 1024
    assert all(lo <= coordinate <= hi
               for (lo, hi), coordinate in zip(receipt.domain, receipt.point))


def test_automatic_nonconvex_irrational_minimum_has_a_proved_global_value_gap():
    x = s.Symbol("global_well_x")
    source = (x**2 - 2)**2
    program = s.compile_objective(source, x)
    tolerance = Fraction(1, 1000)
    result = program.optimize_box_certified([(-2, 2)], tolerance=tolerance, precision=16)
    assert type(result) is s.BoxOptimizationReceipt
    assert result.objective == source
    assert result.variables[0].identity_hex == _symbol_binding(x).identity_hex
    assert result.domain == ((-2, 2),)
    assert result.kind == "minimum"
    assert result.precision_digits == 16
    assert_gap(result, tolerance, Fraction(0))
    # +/-sqrt(2) are irrational. The rational candidate is only tolerance
    # optimal, so a positive point value and nonzero certified gap are honest.
    point = rational(result.point[0])
    actual = (point*point - 2)**2
    assert actual > 0
    assert result.point_value == (s.Rational(actual.numerator, actual.denominator),)*2
    assert result.gap > 0
    assert result.leaf_count > 1
    assert "global" in result.claim and "box" in result.claim
    assert not hasattr(result, "uniqueness_proved")


@pytest.mark.parametrize("kind, point, value", [
    ("minimum", (-1, 4), -9),
    ("maximum", (2, -3), 8),
])
def test_boundary_search_respects_kind_and_nested_compiled_variable_order(kind, point, value):
    x, y = s.symbols("global_affine_x global_affine_y")
    source = x - 2*y
    program = s.compile_objective(source, ((x,), y))
    result = program.optimize_box_certified([(-1, 2), (-3, 4)], kind=kind)
    assert result.objective == source
    assert result.point == point
    assert result.value == result.point_value == (value, value)
    assert result.leaf_count == 1
    assert result.precision_digits == 32
    assert_gap(result, Fraction(1, 10000), Fraction(value))


def test_nonconvex_multivariate_minimum_finds_mixed_boundary_corners():
    x, y = s.symbols("global_corner_x global_corner_y")
    source = -(x-y)**2
    program = s.compile_objective(source, (x, y))
    tolerance = Fraction(1, 1000)
    result = program.optimize_box_certified([(-1, 1), (-1, 1)], tolerance=tolerance)
    assert result.point in ((-1, 1), (1, -1))
    assert result.value == (-4, -4)
    assert result.leaf_count > 1
    assert_gap(result, tolerance, Fraction(-4))
    maximum = program.optimize_box_certified(
        [(-1, 1), (-1, 1)], kind="maximum", tolerance=tolerance,
    )
    assert maximum.value == (0, 0)
    assert maximum.leaf_count == 1
    assert_gap(maximum, tolerance, Fraction(0))


def test_multiple_nonconvex_wells_and_flat_directions_are_admitted():
    x, y = s.symbols("global_multiple_x global_multiple_y")
    for source in ((x**2-1)**2 + (y**2-1)**2, (x-y)**2, s.Integer(7)):
        result = s.compile_objective(source, (x, y)).optimize_box_certified(
            [(-2, 2), (-2, 2)],
        )
        expected = 7 if source == 7 else 0
        assert result.value == (expected, expected)
        assert_gap(result, Fraction(1, 10000), Fraction(expected))


def test_transcendental_global_maximum_accepts_a_rational_tolerance_optimal_point():
    x = s.Symbol("global_sine_x")
    source = s.sin(x)
    result = s.compile_objective(source, x).optimize_box_certified(
        [(0, 4)], kind="maximum", tolerance=Fraction(1, 1000), precision=16,
    )
    assert_gap(result, Fraction(1, 1000), Fraction(1))
    assert result.point_value[1] < 1
    assert result.gap > 0
    assert result.leaf_count > 1


@pytest.mark.parametrize("name, domain, expected", [
    ("exp", (0, 1), 1),
    ("log", (1, 4), 0),
])
def test_monotone_elementary_boundary_minimum_needs_one_leaf(name, domain, expected):
    x = s.Symbol("global_elementary_x")
    result = s.compile_objective(getattr(s, name)(x), x).optimize_box_certified(
        [domain], precision=16,
    )
    assert result.point == (domain[0],)
    assert result.leaf_count == 1
    assert_gap(result, Fraction(1, 10000), Fraction(expected))


def test_fixed_coordinates_singleton_boxes_and_sixteen_variable_flat_objective():
    x, y = s.symbols("global_fixed_x global_fixed_y")
    result = s.compile_objective(x*x+y, (x, y)).optimize_box_certified(
        [(2, 2), (-1, 1)],
    )
    assert result.point == (2, -1)
    assert result.value == (3, 3)
    point = s.compile_objective(x*x, x).optimize_box_certified(
        [(Fraction(1, 3), Fraction(1, 3))],
    )
    assert point.point == (s.Rational(1, 3),)
    assert point.value == (s.Rational(1, 9),)*2
    assert point.leaf_count == 1
    variables = tuple(s.Symbol(f"global_flat_{index}") for index in range(16))
    flat = s.compile_objective(s.Integer(7), variables).optimize_box_certified(
        [(-1, 1)]*16,
    )
    assert flat.point == (0,)*16
    assert flat.value == (7, 7)
    assert flat.leaf_count == 1


def test_exact_tiny_values_and_tolerances_do_not_round_through_binary64():
    x = s.Symbol("global_tiny_x")
    tiny = s.Rational(1, 10**400)
    tolerance = Fraction(1, 10**500)
    result = s.compile_objective(tiny*x, x).optimize_box_certified(
        [(0, 1)], tolerance=tolerance,
    )
    assert result.point == (0,)
    assert result.value == (0, 0)
    assert result.leaf_count == 1
    assert_gap(result, tolerance, Fraction(0))


def test_subdivision_recovers_interval_dependency_in_the_held_original_denominator():
    x = s.Symbol("global_dependency_x")
    denominator = s.Add(1, s.Mul(x, x, evaluate=False), evaluate=False)
    source = s.Pow(denominator, -1, evaluate=False)
    result = s.compile_objective(source, x).optimize_box_certified([(-1, 1)])
    assert result.objective == source
    assert result.value == (s.Rational(1, 2),)*2
    assert result.point in ((-1,), (1,))
    assert result.leaf_count > 1


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_unresolved_domain_cells_block_acceptance_despite_exact_good_values_elsewhere(hidden):
    x = s.Symbol("global_domain_x")
    inverse = s.Pow(s.Add(x, 1, evaluate=False), -1, evaluate=False)
    source = (s.Mul(0, inverse, evaluate=False) if hidden == "zero_product"
              else s.Pow(inverse, 0, evaluate=False))
    program = s.compile_objective(source, x)
    with pytest.raises(ValueError):
        program.optimize_box_certified([(-1, 1)], max_leaves=8)


def test_captured_assumptions_and_typed_variable_identity_remain_bound_to_the_box():
    x = s.Symbol("global_positive_x", positive=True)
    binding = _symbol_binding(x)
    program = s.compile_objective((x-1)**2, x)
    original = dict(x._assumptions)
    try:
        x._assumptions.clear()
        x._assumptions["negative"] = True
        result = program.optimize_box_certified([(Fraction(1, 2), 2)])
        assert result.variables[0].identity_hex == binding.identity_hex
        assert_gap(result, Fraction(1, 10000), Fraction(0))
        with pytest.raises(ValueError, match="positive"):
            program.optimize_box_certified([(-1, 2)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(original)


@pytest.mark.parametrize("value", [True, 0.001, "1/1000", s.Float(0.001)])
def test_tolerance_and_bounds_require_explicit_exact_values(value):
    x = s.Symbol("global_exact_x")
    program = s.compile_objective(x*x, x)
    with pytest.raises(TypeError, match="exact"):
        program.optimize_box_certified([(-1, 1)], tolerance=value)
    with pytest.raises(TypeError, match="exact"):
        program.optimize_box_certified([(value, 1)])


@pytest.mark.parametrize("tolerance", [0, -1, Fraction(-1, 1000)])
def test_nonpositive_exact_tolerance_is_refused(tolerance):
    x = s.Symbol("global_tolerance_x")
    with pytest.raises(ValueError, match="positive exact rational"):
        s.compile_objective(x*x, x).optimize_box_certified([(-1, 1)], tolerance=tolerance)


@pytest.mark.parametrize("max_leaves", [0, 1025, 10**100, True, 1.0])
def test_search_leaf_limit_is_bounded_and_closed_to_custom_conversion(max_leaves):
    x = s.Symbol("global_limits_x")
    with pytest.raises(ValueError, match="max_leaves"):
        s.compile_objective(x*x, x).optimize_box_certified([(-1, 1)], max_leaves=max_leaves)


def test_failed_accuracy_produces_no_local_or_partial_success():
    x = s.Symbol("global_insufficient_x")
    program = s.compile_objective((x*x-2)**2, x)
    budget = ample_budget()
    before = budget.remaining_steps
    with pytest.raises(ValueError, match="gap|tolerance"):
        program.optimize_box_certified([(-2, 2)], max_leaves=1, budget=budget)
    assert budget.remaining_steps < before


def test_inexact_source_oversized_rationals_and_invalid_controls_fail_before_search():
    x = s.Symbol("global_bad_source_x")
    source = s.Add(x*x, s.Mul(s.Float(0.5), x, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        s.compile_objective(source, x).optimize_box_certified([(-1, 1)])
    program = s.compile_objective(x*x, x)
    with pytest.raises(ValueError, match="4096-bit"):
        program.optimize_box_certified([(-1, 1)], tolerance=Fraction(1, 1 << 4096))
    for precision in (7, 129, True):
        with pytest.raises(ValueError, match="precision"):
            program.optimize_box_certified([(-1, 1)], precision=precision)
    with pytest.raises(ValueError, match="kind"):
        program.optimize_box_certified([(-1, 1)], kind="stationary")
    with pytest.raises(TypeError, match="string"):
        program.optimize_box_certified([(-1, 1)], kind=1)
    with pytest.raises(ValueError, match="lower bound"):
        program.optimize_box_certified([(1, -1)])
    with pytest.raises(TypeError, match="entries"):
        program.optimize_box_certified([])


def test_admission_does_not_execute_user_numeric_sequence_or_string_callbacks():
    x = s.Symbol("global_callbacks_x")
    program = s.compile_objective(x*x, x)
    calls = []

    class Numeric:
        def __float__(self):
            calls.append("float")
            return 0.001

        def __int__(self):
            calls.append("int")
            return 1

    class CustomList(list):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

    class CustomString(str):
        def __eq__(self, other):
            calls.append("eq")
            return super().__eq__(other)

    class CustomInt(int):
        def __int__(self):
            calls.append("custom_int")
            return super().__int__()

    for value in (Numeric(), CustomInt(1)):
        with pytest.raises(TypeError):
            program.optimize_box_certified([(-1, 1)], tolerance=value)
    with pytest.raises(TypeError):
        program.optimize_box_certified(CustomList([(-1, 1)]))
    with pytest.raises(TypeError):
        program.optimize_box_certified([(-1, 1)], kind=CustomString("minimum"))
    with pytest.raises(ValueError):
        program.optimize_box_certified([(-1, 1)], max_leaves=CustomInt(4))
    assert calls == []


def test_accepted_receipt_owns_its_inputs_and_cannot_be_constructed_or_reinitialized():
    x = s.Symbol("global_receipt_x")
    domain = [[-1, 1]]
    receipt = s.compile_objective(x*x, x).optimize_box_certified(domain)
    domain[0][:] = [100, 101]
    assert receipt.domain == ((-1, 1),)
    assert receipt.point == (0,)
    assert type(receipt.domain) is tuple and type(receipt.point) is tuple
    with pytest.raises(AttributeError):
        receipt.gap = 100
    with pytest.raises(AttributeError):
        del receipt._receipt
    with pytest.raises(AttributeError):
        receipt.__init__(receipt._receipt)
    with pytest.raises(TypeError):
        s.BoxOptimizationReceipt(object())
    with pytest.raises(TypeError):
        _native.BoxOptimizationReceipt()
    with pytest.raises(AttributeError):
        receipt._receipt.point = (100,)
    from sympy.utilities import BoxOptimizationReceipt
    assert BoxOptimizationReceipt is s.BoxOptimizationReceipt


def test_native_entry_checks_exact_tolerance_and_publishes_only_accepted_receipts():
    x = s.Symbol("global_native_x")
    native = s.compile_objective(x*x, x).native_objective
    domain = [(_native_expr(-1), _native_expr(1))]
    with pytest.raises(TypeError):
        native.optimize_box(domain, 0.001)
    with pytest.raises(ValueError):
        native.optimize_box(domain, _native_expr(-1))
    with pytest.raises(ValueError):
        native.optimize_box(domain, _native_expr(x))
    result = native.optimize_box(domain, _native_expr(s.Rational(1, 1000)))
    assert type(result) is _native.BoxOptimizationReceipt
    assert result.kind == "minimum"
    assert result.leaf_count == 1


def test_generation_and_independent_verification_share_one_cumulative_budget():
    x = s.Symbol("global_budget_x")
    program = s.compile_objective(x, x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.optimize_box_certified(
            [(-1, 1)], budget=ample_budget(max_memory_bytes=100*1024*1024),
        )
    probe = ample_budget()
    program.optimize_box_certified([(-1, 1)], budget=probe)
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 128*1024*1024
    exact = ample_budget(max_memory_bytes=cost)
    program.optimize_box_certified([(-1, 1)], budget=exact)
    assert exact.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.optimize_box_certified([(-1, 1)], budget=exact)


def test_cancelled_or_exhausted_budgets_and_last_checkpoint_cannot_publish(monkeypatch):
    x = s.Symbol("global_cancel_x")
    program = s.compile_objective(x*x, x)
    budget = ample_budget()
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.optimize_box_certified([(-1, 1)], budget=budget)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.optimize_box_certified([(-1, 1)], budget=ample_budget(max_steps=0))
    module = importlib.import_module("sympy.utilities._box_optimization")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_native_acceptance(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_native_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        program.optimize_box_certified([(-1, 1)], budget=budget)


def test_unchecked_native_result_is_not_accepted_by_the_public_adapter(monkeypatch):
    x = s.Symbol("global_forged_x")
    module = importlib.import_module("sympy.utilities._box_optimization")
    monkeypatch.setattr(module, "_call_budgeted", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="accepted native"):
        s.compile_objective(x*x, x).optimize_box_certified([(-1, 1)])
