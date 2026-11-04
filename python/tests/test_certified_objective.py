"""Original-source second-order bounds and local stationary-point evidence."""

from fractions import Fraction
import importlib

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_box_enclosure_matches_independent_polynomial_derivatives():
    x, y = s.symbols("objective_box_x objective_box_y")
    source = 2*x*x + x*y + 3*y*y
    result = s.compile_objective(source, (x, y)).enclose([(1, 2), (3, 4)])
    assert type(result) is s.ObjectiveEnclosure
    assert result.objective == source
    assert result.domain == ((1, 2), (3, 4))
    assert result.precision_digits == 32
    assert [variable.name for variable in result.variables] == [x.name, y.name]
    lower, upper = result.value
    assert lower <= 32 <= 64 <= upper
    assert result.gradient == ((7, 12), (19, 26))
    assert result.hessian == (((4, 4), (1, 1)), ((1, 1), (6, 6)))


def test_exact_point_boxes_do_not_pass_through_binary64():
    x = s.Symbol("objective_large_point_x")
    huge = 10**400
    result = s.compile_objective(x*x, x).enclose([(huge, huge)])
    assert result.value == (huge*huge, huge*huge)
    assert result.gradient == ((2*huge, 2*huge),)
    assert result.hessian == (((2, 2),),)
    third = s.compile_objective(x*x, x).enclose([(Fraction(1, 3), s.Rational(1, 3))])
    assert third.value == (s.Rational(1, 9), s.Rational(1, 9))


def test_strict_minimum_has_checked_source_coordinates_and_local_claim():
    x, y = s.symbols("objective_minimum_x objective_minimum_y")
    source = (x-1)**2 + 2*(y+2)**2
    program = s.compile_objective(source, ((x,), y))
    result = program.certify_stationary_point([(0, 2), (-3, -1)])
    assert type(result) is s.StationaryPointReceipt
    assert result.kind == "strict_minimum"
    assert result.objective == source
    assert result.stationary_enclosure == ((1, 1), (-2, -2))
    assert result.center == (1, -2)
    assert result.preconditioner == ((s.Rational(1, 2), 0), (0, s.Rational(1, 4)))
    assert result.hessian_enclosure == (((2, 2), (0, 0)), ((0, 0), (4, 4)))
    assert result.contraction_bound == 0
    assert "declared closed box" in result.claim
    lower, upper = result.objective_enclosure
    assert lower <= 0 <= upper
    assert program.evaluate((1,), -2) == (0.0, [0.0, 0.0], [[2.0, 0.0], [0.0, 4.0]])


@pytest.mark.parametrize("kind", ["strict_maximum", "saddle", "unclassified"])
def test_curvature_classification_does_not_overstate_inconclusive_pivots(kind):
    x, y = s.symbols("objective_curvature_x objective_curvature_y")
    source = {
        "strict_maximum": -x*x-2*y*y,
        "saddle": x*x-y*y,
        # Hessian [[0,1],[1,0]] is nonsingular; the sufficient LDL test cannot
        # classify it without pivoting. Stationarity remains independently proved.
        "unclassified": x*y,
    }[kind]
    result = s.compile_objective(source, (x, y)).certify_stationary_point([(-1, 1)]*2)
    assert result.kind == kind
    assert result.stationary_enclosure == ((0, 0), (0, 0))
    assert result.contraction_bound == 0


def test_transcendental_stationary_point_uses_original_interval_derivatives():
    x = s.Symbol("objective_exponential_x")
    result = s.compile_objective(s.exp(x)-x, x).certify_stationary_point(
        [(s.Rational(-1, 10), s.Rational(1, 10))], precision=16,
    )
    lo, hi = result.stationary_enclosure[0]
    assert s.Rational(-1, 10) < lo <= 0 <= hi < s.Rational(1, 10)
    assert result.kind == "strict_minimum"
    assert 0 <= result.contraction_bound < 1
    assert result.objective_enclosure[0] <= 1 <= result.objective_enclosure[1]
    assert result.hessian_enclosure[0][0][0] > 0


def test_dynamic_positive_power_has_independent_second_order_enclosures():
    x = s.Symbol("objective_dynamic_power_x")
    program = s.compile_objective(x**x, x)
    result = program.enclose([(2, 2)], precision=16)
    assert result.value[0] <= 4 <= result.value[1]
    # f'(2)=4*(log(2)+1); f''(2)=4*((log(2)+1)**2+1/2).
    assert 6 < result.gradient[0][0] <= result.gradient[0][1] < 8
    assert 12 < result.hessian[0][0][0] <= result.hessian[0][0][1] < 16
    with pytest.raises(ValueError):
        program.enclose([(-1, 1)])


@pytest.mark.parametrize("case", ["absent", "multiple", "singular", "boundary"])
def test_invalid_stationary_point_claims_refuse(case):
    x = s.Symbol("objective_invalid_" + case)
    source, box = {
        "absent": (x, [(-1, 1)]),
        "multiple": (x**4-x*x, [(-2, 2)]),
        "singular": (x**4, [(-1, 1)]),
        "boundary": (x*x, [(0, 1)]),
    }[case]
    with pytest.raises(ValueError):
        s.compile_objective(source, x).certify_stationary_point(box, preconditioner=[[1]])


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_complete_original_c2_domain_cannot_be_erased(hidden):
    x = s.Symbol("objective_hidden_domain_" + hidden)
    hidden_source = (s.Mul(0, s.log(x), evaluate=False) if hidden == "zero_product"
                     else s.Pow(s.log(x), 0, evaluate=False))
    source = s.Add((x-1)**2, hidden_source, evaluate=False)
    program = s.compile_objective(source, x)
    for operation in (program.enclose, program.certify_stationary_point):
        with pytest.raises(ValueError):
            operation([(-1, 2)])


@pytest.mark.parametrize("power", [s.Rational(1, 2), s.Rational(3, 2)])
def test_second_derivative_domain_is_checked_even_when_value_is_finite(power):
    x = s.Symbol("objective_c2_x")
    program = s.compile_objective(x**power, x)
    with pytest.raises(ValueError):
        program.enclose([(0, 1)])
    assert program.enclose([(1, 1)]).value == (1, 1)


def test_manual_witness_snapshots_and_false_preconditioners():
    x = s.Symbol("objective_manual_x")
    program = s.compile_objective((x-1)**2, x)
    box, center, preconditioner = [[0, 2]], [1], [[s.Rational(1, 2)]]
    result = program.certify_stationary_point(box, center=center, preconditioner=preconditioner)
    box[0][0], center[0], preconditioner[0][0] = 100, 100, 0
    assert result.domain == ((0, 2),)
    assert result.center == (1,)
    assert result.preconditioner == ((s.Rational(1, 2),),)
    with pytest.raises(ValueError):
        program.certify_stationary_point([(0, 2)], preconditioner=[[0]])
    with pytest.raises(ValueError, match="inside"):
        program.certify_stationary_point([(0, 2)], center=[3])


@pytest.mark.parametrize("value", [0.5, True, "1/2", s.Float(0.5)])
def test_inexact_endpoints_and_witnesses_are_rejected(value):
    x = s.Symbol("objective_invalid_exact_input_x")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(TypeError):
        program.enclose([(value, 2)])
    with pytest.raises(TypeError):
        program.certify_stationary_point([(0, 2)], center=[value])
    with pytest.raises(TypeError):
        program.certify_stationary_point([(0, 2)], preconditioner=[[value]])


def test_inexact_source_is_not_promoted_by_exact_witness_inputs():
    x = s.Symbol("objective_float_source_x")
    source = s.Add(s.Mul(s.Float(0.5), x*x, evaluate=False), -x, evaluate=False)
    program = s.compile_objective(source, x)
    assert program.evaluate(1) == (-0.5, [0.0], [[1.0]])
    for operation in (program.enclose, program.certify_stationary_point):
        with pytest.raises(ValueError, match="exact source"):
            operation([(0, 2)])


def test_typed_coordinates_and_captured_assumptions_remain_distinct():
    plain = s.Symbol("objective_typed_x")
    positive = s.Symbol("objective_typed_x", positive=True)
    left, right = s.Dummy("objective_x"), s.Dummy("objective_x")
    variables = (plain, positive, left, right)
    source = sum((variable-index-1)**2 for index, variable in enumerate(variables))
    program = s.compile_objective(source, variables)
    binding = _symbol_binding(positive)
    facts = dict(positive._assumptions)
    try:
        positive._assumptions.clear()
        result = program.certify_stationary_point([(0, 2), (1, 3), (2, 4), (3, 5)])
        assert result.stationary_enclosure == ((1, 1), (2, 2), (3, 3), (4, 4))
        assert result.variables[0].plain
        assert result.variables[1].identity_hex == binding.identity_hex
        assert result.variables[2].name != result.variables[3].name
        with pytest.raises(ValueError, match="positive"):
            program.enclose([(0, 2), (-1, 3), (2, 4), (3, 5)])
    finally:
        positive._assumptions.update(facts)


def test_constructor_mutation_cannot_change_captured_assumptions(monkeypatch):
    x = s.Symbol("objective_constructor_snapshot_x", positive=True)
    facts = dict(x._assumptions)
    constructor = _native.VerifiedObjective

    def mutate_after_construction(source, variables):
        result = constructor(source, variables)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(_native, "VerifiedObjective", mutate_after_construction)
    try:
        program = s.compile_objective((x-1)**2, x)
        assert program.enclose([(1, 2)]).domain == ((1, 2),)
        with pytest.raises(ValueError, match="positive"):
            program.enclose([(-2, -1)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_inconsistent_binding_snapshot_disables_only_certification(monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    x = s.Symbol("objective_binding_snapshot_x", positive=True)
    source, facts = (x-1)**2, dict(x._assumptions)
    binding_function = module._symbol_binding

    def mutate_after_binding(variable):
        binding = binding_function(variable)
        variable._assumptions.clear()
        variable._assumptions["negative"] = True
        return binding

    monkeypatch.setattr(module, "_symbol_binding", mutate_after_binding)
    try:
        program = s.compile_objective(source, x)
        assert program.evaluate(1) == (0.0, [0.0], [[2.0]])
        with pytest.raises(ValueError, match="captured symbol bindings"):
            program.enclose([(0, 2)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_receipts_are_immutable_and_have_no_unchecked_constructor():
    x = s.Symbol("objective_immutable_x")
    program = s.compile_objective((x-1)**2, x)
    for receipt in (program.enclose([(0, 2)]), program.certify_stationary_point([(0, 2)])):
        with pytest.raises(AttributeError):
            receipt.objective = x
        with pytest.raises(AttributeError):
            del receipt._receipt
        with pytest.raises(AttributeError):
            receipt.__init__(receipt._receipt)
        with pytest.raises(AttributeError):
            receipt._receipt.domain = [(100, 200)]
    for cls in (s.ObjectiveEnclosure, s.StationaryPointReceipt,
                _native.ObjectiveEnclosure, _native.StationaryPointReceipt):
        with pytest.raises(TypeError):
            cls({"verified": True})


def test_one_budget_spans_approximation_exact_proposal_and_acceptance():
    x = s.Symbol("objective_shared_budget_x")
    program = s.compile_objective((x-1)**2, x)
    budget = ample_budget()
    assert program.evaluate(0, budget=budget)[0] == 1
    after_numeric = budget.remaining_steps
    assert program.enclose([(0, 2)], budget=budget).hessian == (((2, 2),),)
    after_enclosure = budget.remaining_steps
    result = program.certify_stationary_point([(0, 2)], budget=budget)
    assert result.stationary_enclosure == ((1, 1),)
    assert budget.remaining_steps < after_enclosure < after_numeric < 1 << 45
    assert (1 << 40)-budget.remaining_memory_bytes >= 3*64*1024*1024


def test_repeated_objective_enclosures_cannot_reset_the_memory_budget():
    x = s.Symbol("objective_cumulative_memory_x")
    program = s.compile_objective(x*x, x)
    probe = ample_budget()
    program.enclose([(0, 2)], budget=probe)
    cost = (1 << 40)-probe.remaining_memory_bytes
    budget = ample_budget(max_memory_bytes=cost)
    assert program.enclose([(0, 2)], budget=budget).hessian == (((2, 2),),)
    assert budget.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.enclose([(0, 2)], budget=budget)
    assert budget.remaining_memory_bytes == 0


def test_cancelled_budget_refuses_every_new_certified_operation():
    x = s.Symbol("objective_cancelled_x")
    program = s.compile_objective((x-1)**2, x)
    budget = ample_budget()
    budget.cancel()
    for operation in (program.enclose, program.certify_stationary_point):
        with pytest.raises(s.EvaluationCancelled):
            operation([(0, 2)], budget=budget)
    assert budget.remaining_steps == 1 << 45


@pytest.mark.parametrize("kind", ["enclosure", "stationary"])
def test_cancellation_after_receipt_assembly_refuses_publication(monkeypatch, kind):
    module = importlib.import_module("sympy.utilities._certified_objective")
    x = s.Symbol("objective_publication_x")
    program = s.compile_objective((x-1)**2, x)
    budget = ample_budget()
    name = "ObjectiveEnclosure" if kind == "enclosure" else "StationaryPointReceipt"
    wrapper, created = getattr(module, name), []

    def cancel_after_wrapper(native):
        receipt = wrapper(native)
        created.append(receipt)
        budget.cancel()
        return receipt

    monkeypatch.setattr(module, name, cancel_after_wrapper)
    with pytest.raises(s.EvaluationCancelled):
        operation = program.enclose if kind == "enclosure" else program.certify_stationary_point
        operation([(0, 2)], budget=budget)
    assert len(created) == 1


def test_operational_inverse_failure_is_not_retried_as_a_mathematical_refusal(monkeypatch):
    module = importlib.import_module("sympy.utilities._certified_objective")
    x = s.Symbol("objective_inverse_budget_x")
    program = s.compile_objective((x-1)**2, x)
    budget, calls = ample_budget(), []

    def exhausted(matrix, supplied):
        calls.append(supplied)
        raise s.EvaluationBudgetExceeded("preconditioner limit")

    monkeypatch.setattr(module, "_bounded_inverse", exhausted)
    with pytest.raises(s.EvaluationBudgetExceeded, match="preconditioner limit"):
        program.certify_stationary_point([(0, 2)], budget=budget)
    assert calls == [budget]
    assert budget.remaining_steps < 1 << 45


def test_bad_dimensions_precision_storage_and_budget_objects_refuse():
    x = s.Symbol("objective_limits_x")
    program = s.compile_objective((x-1)**2, x)
    for precision in (7, 129, True, 32.0):
        with pytest.raises(ValueError, match="precision"):
            program.enclose([(0, 2)], precision=precision)
    with pytest.raises(ValueError, match="positive width"):
        program.certify_stationary_point([(1, 1)])
    with pytest.raises(ValueError, match="bit limit"):
        program.enclose([(0, 1 << 4097)])
    with pytest.raises(TypeError):
        program.enclose([])
    for variables in ((), tuple(s.Symbol(f"objective_limit_{i}") for i in range(17))):
        with pytest.raises(ValueError, match="1..16"):
            s.compile_objective(7, variables).enclose([(0, 2)]*len(variables))

    class Hostile:
        def _checkpoint(self):
            raise AssertionError("a budget lookalike must not execute")

    for operation in (program.enclose, program.certify_stationary_point):
        with pytest.raises(TypeError, match="EvaluationBudget"):
            operation([(0, 2)], budget=Hostile())
    native_box = [(_native.py_rational(0, 1), _native.py_rational(2, 1))]
    budget = ample_budget(max_memory_bytes=0)
    before = budget.remaining_steps, budget.remaining_allocations
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.native_objective.enclose(native_box, budget=budget)
    assert (budget.remaining_steps, budget.remaining_allocations) == before


def test_objective_receipts_are_exported_from_both_public_namespaces():
    from sympy.utilities import ObjectiveEnclosure, StationaryPointReceipt
    assert ObjectiveEnclosure is s.ObjectiveEnclosure
    assert StationaryPointReceipt is s.StationaryPointReceipt
