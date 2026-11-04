"""Original-source global box extrema, including boundary and flat cases."""

from fractions import Fraction
import importlib

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_global_box_minimum_retains_original_source_and_exact_value():
    x, y = s.symbols("box_minimum_x box_minimum_y")
    source = (x + 1)**2 + 2*(y - 2)**2
    program = s.compile_objective(source, ((x,), y))
    receipt = program.certify_box_extremum([(0, 3), (0, 4)], [0, 2])
    assert type(receipt) is s.BoxExtremumReceipt
    assert receipt.objective == source
    assert receipt.point == (0, 2)
    assert receipt.domain == ((0, 3), (0, 4))
    assert receipt.kind == "minimum"
    assert receipt.value == (1, 1)
    assert receipt.gradient == ((2, 2), (0, 0))
    assert receipt.hessian == (((2, 2), (0, 0)), ((0, 0), (4, 4)))
    assert receipt.uniqueness_proved is True
    assert receipt.precision_digits == 32
    assert receipt.claim == "global minimum on the declared closed box"


def test_global_maximum_reverses_curvature_and_first_order_conditions():
    x = s.Symbol("box_maximum_x")
    source = 5 - (x - 1)**2
    program = s.compile_objective(source, x)
    receipt = program.certify_box_extremum([(-3, 4)], [1], kind="maximum")
    assert receipt.kind == "maximum"
    assert receipt.value == (5, 5)
    assert receipt.uniqueness_proved
    with pytest.raises(ValueError, match="semidefiniteness"):
        program.certify_box_extremum([(-3, 4)], [1], kind="minimum")


def test_affine_corner_has_global_extrema_and_strict_supporting_uniqueness():
    x, y = s.symbols("box_affine_x box_affine_y")
    program = s.compile_objective(2*x - 3*y, (x, y))
    domain = [(-1, 2), (-2, 3)]
    minimum = program.certify_box_extremum(domain, [-1, 3])
    maximum = program.certify_box_extremum(domain, [2, -2], kind="maximum")
    assert minimum.value == (-11, -11)
    assert maximum.value == (10, 10)
    assert minimum.uniqueness_proved and maximum.uniqueness_proved
    assert minimum.hessian == (((0, 0), (0, 0)), ((0, 0), (0, 0)))
    for wrong in ([2, 3], [-1, -2], [0, 0]):
        with pytest.raises(ValueError, match="first-order"):
            program.certify_box_extremum(domain, wrong)


@pytest.mark.parametrize("case", ["constant", "free_direction", "quartic", "coupled"])
def test_semidefinite_curvature_does_not_invent_uniqueness(case):
    x, y = s.symbols("box_flat_x box_flat_y")
    source = {"constant": s.Integer(7), "free_direction": x*x,
              "quartic": x**4, "coupled": (x+y)**2}[case]
    receipt = s.compile_objective(source, (x, y)).certify_box_extremum(
        [(-1, 1), (-1, 1)], [0, 0],
    )
    expected = 7 if case == "constant" else 0
    assert receipt.value == (expected, expected)
    assert receipt.uniqueness_proved is False
    assert receipt.kind == "minimum"


@pytest.mark.parametrize("case", ["indefinite_zero_pivot", "maximum", "double_well"])
def test_false_global_minimum_claims_fail_even_at_stationary_points(case):
    x, y = s.symbols("box_bad_curvature_x box_bad_curvature_y")
    source = {"indefinite_zero_pivot": x*y, "maximum": -x*x-y*y,
              "double_well": (x*x-1)**2+y*y}[case]
    with pytest.raises(ValueError, match="semidefiniteness"):
        s.compile_objective(source, (x, y)).certify_box_extremum(
            [(-2, 2), (-2, 2)], [0, 0],
        )


def test_nondiagonally_dominant_positive_hessian_is_supported():
    x, y = s.symbols("box_coupled_x box_coupled_y")
    source = (x + 2*y)**2 + y*y
    receipt = s.compile_objective(source, (x, y)).certify_box_extremum(
        [(-100, 100), (-100, 100)], [0, 0],
    )
    assert receipt.value == (0, 0)
    assert receipt.hessian == (((2, 2), (4, 4)), ((4, 4), (10, 10)))
    assert receipt.uniqueness_proved


def test_transcendental_boundary_minimum_and_fixed_coordinates():
    x, y = s.symbols("box_transcendental_x box_fixed_y")
    program = s.compile_objective(s.exp(x) + y, (x, y))
    receipt = program.certify_box_extremum([(0, 1), (3, 3)], [0, 3], precision=16)
    assert receipt.value[0] <= 4 <= receipt.value[1]
    assert receipt.uniqueness_proved
    assert receipt.point == (0, 3)
    assert receipt.gradient[0][0] > 0


def test_exact_fractions_and_tiny_curvature_never_round_through_binary64():
    x = s.Symbol("box_tiny_x")
    tiny = s.Rational(1, 10**400)
    point = s.Rational(1, 3)
    source = tiny*(x-point)**2
    receipt = s.compile_objective(source, x).certify_box_extremum(
        [(0, 1)], [Fraction(1, 3)],
    )
    assert receipt.point == (point,)
    assert receipt.value == (0, 0)
    assert receipt.hessian == (((2*tiny, 2*tiny),),)
    assert receipt.uniqueness_proved


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_original_domain_is_checked_everywhere_not_only_at_the_candidate(hidden):
    x = s.Symbol("box_original_domain_x")
    partial = s.Pow(x, -1, evaluate=False)
    hole = (s.Mul(0, partial, evaluate=False) if hidden == "zero_product"
            else s.Pow(partial, 0, evaluate=False))
    source = s.Add((x-1)**2, hole, evaluate=False)
    program = s.compile_objective(source, x)
    with pytest.raises(ValueError):
        program.certify_box_extremum([(-1, 2)], [1])


def test_false_candidate_outside_box_and_nonzero_interior_gradient_refuse():
    x = s.Symbol("box_false_point_x")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(ValueError, match="inside"):
        program.certify_box_extremum([(0, 2)], [3])
    with pytest.raises(ValueError, match="first-order"):
        program.certify_box_extremum([(0, 2)], [Fraction(1, 2)])
    with pytest.raises(ValueError, match="first-order"):
        program.certify_box_extremum([(0, 2)], [0])


def test_boundaries_keep_captured_typed_assumptions():
    x = s.Symbol("box_positive_x", positive=True)
    binding = _symbol_binding(x)
    program = s.compile_objective((x-1)**2, x)
    original = dict(x._assumptions)
    try:
        x._assumptions.clear()
        x._assumptions["negative"] = True
        receipt = program.certify_box_extremum([(Fraction(1, 2), 2)], [1])
        assert receipt.variables[0].identity_hex == binding.identity_hex
        assert receipt.value == (0, 0)
        with pytest.raises(ValueError, match="positive"):
            program.certify_box_extremum([(-1, 2)], [1])
    finally:
        x._assumptions.clear()
        x._assumptions.update(original)


@pytest.mark.parametrize("value", [True, 0.5, "1/2", s.Float(0.5)])
def test_approximate_or_custom_point_values_cannot_become_exact_certificates(value):
    x = s.Symbol("box_exact_inputs_x")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(TypeError):
        program.certify_box_extremum([(0, 2)], [value])
    with pytest.raises(TypeError):
        program.certify_box_extremum([(value, 2)], [1])


def test_inexact_source_and_invalid_controls_refuse():
    x = s.Symbol("box_inexact_source_x")
    source = s.Add(x*x, s.Mul(s.Float(0.5), x, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        s.compile_objective(source, x).certify_box_extremum([(-1, 1)], [0])
    program = s.compile_objective(x*x, x)
    for precision in (7, 129, True):
        with pytest.raises(ValueError, match="precision"):
            program.certify_box_extremum([(-1, 1)], [0], precision=precision)
    with pytest.raises(ValueError, match="kind"):
        program.certify_box_extremum([(-1, 1)], [0], kind="stationary")
    with pytest.raises(TypeError, match="string"):
        program.certify_box_extremum([(-1, 1)], [0], kind=1)
    with pytest.raises((TypeError, ValueError)):
        program.certify_box_extremum([(-1, 1)], [])


def test_receipts_own_their_inputs_and_reject_unchecked_construction():
    x = s.Symbol("box_receipt_x")
    program = s.compile_objective(x*x, x)
    domain, point = [[-1, 1]], [0]
    receipt = program.certify_box_extremum(domain, point)
    domain[0][0], point[0] = 100, 100
    assert receipt.domain == ((-1, 1),)
    assert receipt.point == (0,)
    with pytest.raises(AttributeError):
        receipt.point = (100,)
    with pytest.raises(AttributeError):
        receipt.__init__(receipt._receipt)
    with pytest.raises(TypeError):
        s.BoxExtremumReceipt(object())
    with pytest.raises(TypeError):
        _native.BoxExtremumReceipt()
    from sympy.utilities import BoxExtremumReceipt
    assert BoxExtremumReceipt is s.BoxExtremumReceipt


def test_native_boundary_rejects_inexact_witnesses_without_python_adapters():
    x = s.Symbol("box_direct_native_x")
    native = s.compile_objective(x*x, x).native_objective
    domain = [(_native_expr(-1), _native_expr(1))]
    with pytest.raises(TypeError):
        native.certify_box_extremum(domain, [0.0])
    with pytest.raises(ValueError):
        native.certify_box_extremum(domain, [_native_expr(0)], "bad")
    result = native.certify_box_extremum(domain, [_native_expr(0)])
    assert type(result) is _native.BoxExtremumReceipt
    assert result.kind == "minimum"
    assert result.uniqueness_proved


def test_cumulative_budget_is_retained_across_accepted_and_rejected_requests():
    x = s.Symbol("box_budget_x")
    program = s.compile_objective(x*x, x)
    probe = ample_budget()
    program.certify_box_extremum([(-1, 1)], [0], budget=probe)
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 64*1024*1024
    budget = ample_budget(max_memory_bytes=cost)
    program.certify_box_extremum([(-1, 1)], [0], budget=budget)
    assert budget.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.certify_box_extremum([(-1, 1)], [0], budget=budget)
    failed = ample_budget()
    before = failed.remaining_steps
    with pytest.raises(ValueError, match="first-order"):
        program.certify_box_extremum([(-1, 1)], [1], budget=failed)
    assert failed.remaining_steps < before


def test_cancelled_and_exhausted_budgets_cannot_publish_receipts(monkeypatch):
    x = s.Symbol("box_cancel_x")
    program = s.compile_objective(x*x, x)
    budget = ample_budget()
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.certify_box_extremum([(-1, 1)], [0], budget=budget)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.certify_box_extremum([(-1, 1)], [0], budget=ample_budget(max_steps=0))
    module = importlib.import_module("sympy.utilities._certified_objective")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_acceptance(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        program.certify_box_extremum([(-1, 1)], [0], budget=budget)


def test_custom_number_and_kind_hooks_are_not_executed():
    x = s.Symbol("box_custom_x")
    program = s.compile_objective(x*x, x)
    calls = []

    class Number:
        def __float__(self):
            calls.append("float")
            return 0.0

        def __str__(self):
            calls.append("str")
            return "minimum"

    with pytest.raises(TypeError):
        program.certify_box_extremum([(-1, 1)], [Number()])
    with pytest.raises(TypeError):
        program.certify_box_extremum([(-1, 1)], [0], kind=Number())
    assert calls == []
