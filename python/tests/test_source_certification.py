"""Direct exact-source numerical claims, including stable Piecewise branches."""

import importlib
from fractions import Fraction

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


ENCLOSERS = (s.enclose_residual_system, s.enclose_objective)
CHECKERS = (*ENCLOSERS, s.certify_root_box, s.certify_stationary_point)


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_public_exports_and_exact_polynomial_enclosures_without_compilers(monkeypatch):
    utilities = importlib.import_module("sympy.utilities")
    for function in (*CHECKERS, s.certify_box_extremum, s.optimize_box_certified):
        assert getattr(utilities, function.__name__) is function

    def forbidden(*args, **kwargs):
        raise AssertionError("direct source checking must not compile symbolic derivatives")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    x, y = s.symbols("source_polynomial_x source_polynomial_y")
    residuals = s.enclose_residual_system([x*x+y, x*y], ((x,), y), [(1, 2), (3, 4)])
    assert type(residuals) is s.ResidualJacobianEnclosure
    assert residuals.residual_values[0] == (4, 8)
    assert residuals.jacobian == (((2, 4), (1, 1)), ((3, 4), (1, 2)))
    objective = s.enclose_objective(2*x*x+x*y+3*y*y, ((x,), y), [(1, 2), (3, 4)])
    assert type(objective) is s.ObjectiveEnclosure
    assert objective.gradient == ((7, 12), (19, 26))
    assert objective.hessian == (((4, 4), (1, 1)), ((1, 1), (6, 6)))


def test_exact_source_literals_outside_binary64_range_remain_usable():
    x = s.Symbol("source_huge_x")
    huge = 10**400
    residual = x-huge
    root = s.certify_root_box(residual, x, [(huge-1, huge+1)])
    assert root.residuals == (residual,)
    assert root.root_enclosure == ((huge, huge),)
    assert root.contraction_bound == 0
    objective = s.enclose_objective(huge+x*x, x, [(0, 0)])
    assert objective.value == (huge, huge)
    assert objective.gradient == ((0, 0),)
    assert objective.hessian == (((2, 2),),)
    third = s.enclose_objective(x*x, x, [(Fraction(1, 3), s.Rational(1, 3))])
    assert third.value == (s.Rational(1, 9), s.Rational(1, 9))


def test_piecewise_root_retains_original_source_and_skips_inactive_logarithm():
    x = s.Symbol("source_piecewise_root_x")
    source = s.Piecewise((x-1, x > 0), (s.log(-x), True))
    box = [(s.Rational(1, 2), s.Rational(3, 2))]
    enclosed = s.enclose_residual_system(source, x, box)
    assert enclosed.residual_values == ((s.Rational(-1, 2), s.Rational(1, 2)),)
    assert enclosed.jacobian == (((1, 1),),)
    result = s.certify_root_box(source, x, box)
    assert result.residuals == (source,)
    assert result.root_enclosure == ((1, 1),)
    assert result.domain == tuple(box)
    assert result.preconditioner == ((1,),)
    assert result.contraction_bound == 0
    with pytest.raises(ValueError):
        s.compile_residual_system(source, x)


def test_piecewise_branch_selection_uses_exact_bounds_beyond_float_resolution():
    x = s.Symbol("source_exact_branch_x")
    large = 2**53
    source = s.Piecewise((x-large-1, x > large), (1, True))
    box = [(large+s.Rational(1, 2), large+s.Rational(3, 2))]
    result = s.certify_root_box(source, x, box)
    assert result.root_enclosure == ((large+1, large+1),)
    assert result.residuals == (source,)
    assert result.contraction_bound == 0


def test_piecewise_stationary_point_has_local_second_order_evidence():
    x = s.Symbol("source_piecewise_stationary_x")
    source = s.Piecewise(((x-2)**2, x > 0), (s.sqrt(-x), True))
    enclosed = s.enclose_objective(source, x, [(1, 3)])
    assert enclosed.gradient == ((-2, 2),)
    assert enclosed.hessian == (((2, 2),),)
    result = s.certify_stationary_point(source, x, [(1, 3)])
    assert result.objective == source
    assert result.stationary_enclosure == ((2, 2),)
    assert result.kind == "strict_minimum"
    assert result.preconditioner == ((s.Rational(1, 2),),)
    assert result.contraction_bound == 0
    with pytest.raises(ValueError):
        s.compile_objective(source, x)


def test_nested_piecewise_preserves_first_matching_branch_inside_arithmetic():
    x = s.Symbol("source_nested_x")
    first = s.Piecewise((x*x, x < 3), (x+100, x < 5), (1/x, True))
    nested = s.Piecewise((first+1, x > 0), (s.log(-x), True))
    result = s.enclose_objective(nested, x, [(1, 2)])
    assert result.objective == nested
    assert result.value == (2, 5)
    assert result.gradient == ((2, 4),)
    assert result.hessian == (((2, 2),),)


@pytest.mark.parametrize("condition,box,selected", [
    (lambda x: s.And(x < 0, s.log(-x) < 1), [(1, 2)], False),
    (lambda x: s.Or(x < 0, s.log(x) < -1), [(-2, -1)], True),
    (lambda x: s.Not(s.And(x < 0, x > -2), evaluate=False), [(1, 2)], True),
])
def test_boolean_guards_short_circuit_domain_failures(condition, box, selected):
    x = s.Symbol("source_boolean_x")
    source = s.Piecewise((x*x, condition(x)), (3*x, True))
    result = s.enclose_objective(source, x, box)
    assert result.hessian == (((2 if selected else 0, 2 if selected else 0),),)


@pytest.mark.parametrize("relation", [s.Lt, s.Le, s.Gt, s.Ge, s.Eq, s.Ne])
@pytest.mark.parametrize("box", [[(-1, 1)], [(0, 0)]])
def test_crossing_and_point_seams_do_not_become_derivative_claims(relation, box):
    x = s.Symbol("source_seam_x")
    source = s.Piecewise((x*x, relation(x, 0, evaluate=False)), (x, True), evaluate=False)
    for operation in ENCLOSERS:
        with pytest.raises(ValueError):
            operation(source, x, box)


@pytest.mark.parametrize("relation,expected", [(s.Eq, 0), (s.Ne, 2)])
def test_equality_guards_decide_strictly_separated_operands(relation, expected):
    x = s.Symbol("source_disjoint_x")
    source = s.Piecewise((x*x, relation(x, 0, evaluate=False)), (x, True), evaluate=False)
    result = s.enclose_objective(source, x, [(1, 2)])
    assert result.hessian == (((expected, expected),),)


def test_structurally_equal_guards_are_stable_but_still_check_operand_domain():
    x = s.Symbol("source_identity_guard_x")
    source = s.Piecewise((x*x, s.Eq(x, x, evaluate=False)), (x, True), evaluate=False)
    assert s.enclose_objective(source, x, [(-1, 1)]).hessian == (((2, 2),),)
    pole = s.Pow(x, -1, evaluate=False)
    bad = s.Piecewise((x*x, s.Eq(pole, pole, evaluate=False)), (x, True), evaluate=False)
    with pytest.raises(ValueError):
        s.enclose_objective(bad, x, [(-1, 1)])


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_no_matching_piecewise_branch_refuses(operation):
    x = s.Symbol("source_no_match_x")
    source = s.Piecewise((x, x > 0))
    with pytest.raises(ValueError):
        operation(source, x, [(-2, -1)])


@pytest.mark.parametrize("hidden", ["zero_product", "zero_power"])
@pytest.mark.parametrize("operation", ENCLOSERS)
def test_selected_held_domain_obligations_survive_and_inactive_poles_are_lazy(hidden, operation):
    x = s.Symbol("source_held_x")
    value = (s.Mul(0, s.log(x), evaluate=False) if hidden == "zero_product"
             else s.Pow(s.log(x), 0, evaluate=False))
    bad = s.Piecewise((value, True), (x, True), evaluate=False)
    with pytest.raises(ValueError):
        operation(bad, x, [(-2, -1)])
    safe = s.Piecewise((1/x, False), (x*x, True), (s.log(x), True), evaluate=False)
    result = operation(safe, x, [(0, 0)])
    if operation is s.enclose_residual_system:
        assert result.residual_values == ((0, 0),)
    else:
        assert result.value == (0, 0)


def test_typed_symbols_dummies_and_boolean_names_keep_coordinate_identity():
    plain = s.Symbol("source_typed")
    positive = s.Symbol("source_typed", positive=True)
    first, second = s.Dummy("source_dummy"), s.Dummy("source_dummy")
    variables = [plain, positive, first, second]
    source = s.Piecewise((plain+2*positive+3*first+4*second, plain < positive), (0, True))
    result = s.enclose_residual_system(source, variables, [(0, 0), (1, 1), (2, 2), (3, 3)])
    assert result.residual_values == ((20, 20),)
    assert result.jacobian == (((1, 1), (2, 2), (3, 3), (4, 4)),)
    assert result.variables[0].plain
    assert not result.variables[1].plain
    assert result.variables[2].name != result.variables[3].name
    with pytest.raises(ValueError):
        s.enclose_residual_system(source, [plain, first, second], [(0, 0), (2, 2), (3, 3)])

    true_name, false_name = s.Symbol("True"), s.Symbol("False")
    source = s.Piecewise((true_name-false_name, true_name > false_name), (0, True))
    result = s.enclose_residual_system(source, (true_name, false_name), [(3, 4), (1, 2)])
    assert result.residual_values == ((1, 3),)
    assert result.jacobian == (((1, 1), (-1, -1)),)


@pytest.mark.parametrize("condition", [s.Symbol("True"), s.Symbol("False"), s.Dummy("True")])
@pytest.mark.parametrize("operation", ENCLOSERS)
def test_symbolic_boolean_variables_do_not_masquerade_as_literal_guards(condition, operation):
    x = s.Symbol("source_symbolic_guard_x")
    source = s.Piecewise((x*x, condition), (x, True), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        operation(source, (x, condition), [(1, 2), (0, 1)])


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_unknown_inactive_source_and_unbound_coordinates_still_refuse(operation):
    x, y = s.symbols("source_closed_x source_closed_y")
    unbound = s.Piecewise((y, False), (x, True), evaluate=False)
    with pytest.raises(ValueError):
        operation(unbound, x, [(1, 2)])
    opaque = _native.py_function("UnregisteredSourceBranch", _native_expr(x))
    raw = _native.py_function(
        "Piecewise", _native.py_function("Tuple", opaque, _native.Expr("False")),
        _native.py_function("Tuple", _native_expr(x), _native.Expr("True")),
    )
    with pytest.raises((ValueError, NotImplementedError)):
        operation(raw, x, [(1, 2)])


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_changed_relational_hooks_refuse_without_executing_callbacks(operation, monkeypatch):
    x = s.Symbol("source_hook_x")
    source = s.Piecewise((x*x, x > 0), (x, True))
    raw = _native_expr(source)
    effects = []

    def forbidden(cls, *args, **kwargs):
        effects.append("constructor")
        raise AssertionError("source admission must not call changed relational hooks")

    monkeypatch.setattr(s.Gt, "__new__", forbidden)
    for value in (source, raw):
        with pytest.raises(NotImplementedError):
            operation(value, x, [(1, 2)])
    assert effects == []


@pytest.mark.parametrize("operation", CHECKERS)
def test_float_source_is_never_promoted_to_an_exact_certificate(operation):
    x = s.Symbol("source_float_x")
    source = s.Add(x*x, s.Float(0.5), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        operation(source, x, [(-1, 1)])


def test_direct_source_snapshots_matrix_and_assumptions_before_native_checks(monkeypatch):
    x = s.Symbol("source_snapshot_x", positive=True)
    source = x-1
    matrix = s.Matrix([[source]])
    facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    constructor = _native.SourceResidualSystem

    def mutate_after_capture(expressions, variables):
        result = constructor(expressions, variables)
        matrix[0, 0] = 999
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(_native, "SourceResidualSystem", mutate_after_capture)
    try:
        result = s.certify_root_box(matrix, x, [(s.Rational(1, 2), s.Rational(3, 2))])
        assert result.residuals == (source,)
        assert result.root_enclosure == ((1, 1),)
        assert result.variables[0].identity_hex == binding.identity_hex
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_caller_budget_retains_refused_work_and_composes_later_success(operation):
    x = s.Symbol("source_budget_x")
    source = s.Piecewise((x*x, x > 0), (-x, True))
    budget = ample_budget()
    with pytest.raises(ValueError):
        operation(source, x, [(-1, 1)], budget=budget)
    after_failure = budget.remaining_steps
    assert 0 < after_failure < 1 << 45
    operation(source, x, [(1, 2)], budget=budget)
    assert budget.remaining_steps < after_failure
    assert (1 << 40)-budget.remaining_memory_bytes >= 2*64*1024*1024


@pytest.mark.parametrize("operation,source", [
    (s.certify_root_box, lambda x: x-1),
    (s.certify_stationary_point, lambda x: (x-1)**2),
])
def test_private_budget_covers_default_preconditioner_and_final_verification(operation, source, monkeypatch):
    module = importlib.import_module("sympy.utilities._parametric_root")
    budget = ample_budget(max_memory_bytes=96*1024*1024)
    calls = []

    def private_budget():
        calls.append(budget)
        return budget

    monkeypatch.setattr(module, "_fallback_budget", private_budget)
    x = s.Symbol("source_private_budget_x")
    with pytest.raises(s.EvaluationBudgetExceeded):
        operation(source(x), x, [(0, 2)])
    assert calls == [budget]
    assert budget.remaining_memory_bytes < 32*1024*1024
    assert budget.remaining_steps < 1 << 45


@pytest.mark.parametrize("operation", CHECKERS)
def test_cancelled_budget_refuses_before_source_capture(operation, monkeypatch):
    module = importlib.import_module("sympy.utilities.source_certification")
    x = s.Symbol("source_cancel_x")
    source = x*x
    budget = ample_budget()
    budget.cancel()
    before = budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled direct operation must not capture fresh sources")

    monkeypatch.setattr(module, "_residual_context", forbidden)
    monkeypatch.setattr(module, "_objective_context", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        operation(source, x, [(-1, 1)], budget=budget)
    assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


def test_false_witnesses_and_receipt_relabelling_are_rejected():
    x = s.Symbol("source_witness_x")
    with pytest.raises(ValueError):
        s.certify_root_box(x-1, x, [(0, 2)], preconditioner=[[0]])
    with pytest.raises(ValueError):
        s.certify_stationary_point((x-1)**2, x, [(0, 2)], preconditioner=[[0]])
    result = s.certify_root_box(x-1, x, [(0, 2)], center=[1], preconditioner=[[1]])
    with pytest.raises(AttributeError):
        result.domain = ((10, 20),)
    with pytest.raises(AttributeError):
        result.__init__(result._receipt)


def test_native_source_snapshots_enforce_typed_identity_and_complete_guard_grammar():
    x = s.Symbol("source_native_x")
    binding = _symbol_binding(x)
    source = _native_expr(x-1)
    expressions, variables = [source], [binding]
    system = _native.SourceResidualSystem(expressions, variables)
    expressions[0] = _native.py_integer(99)
    variables.clear()
    assert system.residuals == [source]
    assert system.variable_count == system.residual_count == 1
    with pytest.raises(AttributeError):
        system.residuals = []
    with pytest.raises(ValueError):
        _native.SourceResidualSystem([source], [binding, binding])
    fake_boolean = _native_expr(s.Symbol("True", real=True))
    malformed = _native.py_function(
        "Piecewise", _native.py_function("Tuple", source, fake_boolean),
        _native.py_function("Tuple", source, _native.Expr("True")),
    )
    with pytest.raises(ValueError, match="condition"):
        _native.SourceResidualSystem([malformed], [binding])
    with pytest.raises(ValueError, match="condition"):
        _native.SourceObjective(malformed, [binding])


def test_piecewise_global_boundary_minimum_uses_original_source_and_exact_face_signs():
    x, y = s.symbols("source_global_boundary_x source_global_boundary_y")
    source = s.Piecewise(((x+1)**2+y*y, x > 0), (s.log(-x), True))
    result = s.certify_box_extremum(source, ((x,), y), [(1, 3), (-1, 1)], [1, 0])
    assert type(result) is s.BoxExtremumReceipt
    assert result.objective == source
    assert result.domain == ((1, 3), (-1, 1))
    assert result.point == (1, 0)
    assert result.value == (4, 4)
    assert result.gradient == ((4, 4), (0, 0))
    assert result.hessian == (((2, 2), (0, 0)), ((0, 0), (2, 2)))
    assert result.uniqueness_proved
    assert result.claim == "global minimum on the declared closed box"


def test_piecewise_global_maximum_reverses_curvature_and_retains_box_scope():
    x = s.Symbol("source_global_maximum_x")
    source = s.Piecewise((5-(x-2)**2, x > 0), (s.log(-x), True))
    result = s.certify_box_extremum(source, x, [(1, 3)], [2], kind="maximum")
    assert result.kind == "maximum"
    assert result.value == (5, 5)
    assert result.uniqueness_proved
    assert result.objective == source
    assert result.claim == "global maximum on the declared closed box"
    with pytest.raises(ValueError, match="semidefiniteness"):
        s.certify_box_extremum(source, x, [(1, 3)], [2], kind="minimum")


@pytest.mark.parametrize("kind,point", [("minimum", 1), ("maximum", 3)])
def test_piecewise_global_search_finds_boundary_optimizers(kind, point):
    x = s.Symbol("source_global_search_x")
    source = s.Piecewise((x, x > 0), (s.log(-x), True))
    result = s.optimize_box_certified(source, x, [(1, 3)], kind=kind)
    assert type(result) is s.BoxOptimizationReceipt
    assert result.objective == source
    assert result.point == (point,)
    assert result.value == result.point_value == (point, point)
    assert result.gap == 0
    assert result.absolute_tolerance == s.Rational(1, 10000)
    assert result.leaf_count == 1
    assert not hasattr(result, "uniqueness_proved")


def test_piecewise_irrational_optimizer_has_a_proved_global_value_gap():
    x = s.Symbol("source_irrational_optimizer_x")
    source = s.Piecewise(((x*x-2)**2, x > 0), (s.log(-x), True))
    tolerance = Fraction(1, 1000)
    result = s.optimize_box_certified(
        source, x, [(1, 2)], tolerance=tolerance, precision=16,
    )
    assert result.objective == source
    assert result.value[0] <= 0 <= result.value[1]
    assert 0 < result.gap <= s.Rational(1, 1000)
    assert result.value[1]-result.value[0] == result.gap
    point = result.point[0]
    actual = (point*point-2)**2
    assert actual > 0  # A rational witness is not the irrational optimizer.
    assert result.point_value == (actual, actual)
    assert result.value[1] == actual
    assert result.leaf_count > 1
    assert result.absolute_tolerance == s.Rational(1, 1000)


@pytest.mark.parametrize("case", ["outside", "nonstationary", "nonconvex"])
def test_false_or_unproved_global_extremum_points_refuse(case):
    x = s.Symbol("source_false_global_x")
    value, point, box = {
        "outside": ((x-2)**2, [4], [(1, 3)]),
        "nonstationary": ((x-2)**2, [1], [(1, 3)]),
        "nonconvex": (((x-2)**2-1)**2, [2], [(s.Rational(1, 2), s.Rational(7, 2))]),
    }[case]
    source = s.Piecewise((value, x > 0), (s.log(-x), True))
    with pytest.raises(ValueError):
        s.certify_box_extremum(source, x, box, point)


def test_global_source_certifiers_do_not_require_binary64_programs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("exact global checking must not compile derivative programs")

    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    x = s.Symbol("source_global_huge_x")
    huge = 10**400
    source = huge+x
    proved = s.certify_box_extremum(source, x, [(0, 1)], [0])
    searched = s.optimize_box_certified(source, x, [(0, 1)])
    assert proved.value == searched.value == (huge, huge)
    assert searched.point == (0,)
    assert searched.gap == 0


@pytest.mark.parametrize("operation", ["extremum", "optimization"])
def test_global_source_operations_refuse_seams_instead_of_certifying_partial_coverage(operation):
    x = s.Symbol("source_global_seam_x")
    source = s.Piecewise((x*x, x > 0), (x, True))
    with pytest.raises(ValueError):
        if operation == "extremum":
            s.certify_box_extremum(source, x, [(-1, 1)], [0])
        else:
            s.optimize_box_certified(source, x, [(-1, 1)], max_leaves=8)


def test_global_source_operations_share_one_budget_across_proposal_replay_and_prior_calls():
    x = s.Symbol("source_global_budget_x")
    source = s.Piecewise((x, x > 0), (s.log(-x), True))
    budget = ample_budget()
    direct = s.certify_box_extremum(source, x, [(1, 3)], [1], budget=budget)
    after_direct = budget.remaining_steps
    searched = s.optimize_box_certified(source, x, [(1, 3)], budget=budget)
    assert direct.value == searched.value == (1, 1)
    assert budget.remaining_steps < after_direct < 1 << 45
    assert (1 << 40)-budget.remaining_memory_bytes >= 3*64*1024*1024


@pytest.mark.parametrize("operation", ["extremum", "optimization"])
def test_cancelled_global_source_operation_refuses_before_capturing_source(operation, monkeypatch):
    module = importlib.import_module("sympy.utilities.source_certification")
    x = s.Symbol("source_global_cancel_x")
    source = x*x
    budget = ample_budget()
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled global operation must not capture a new source")

    monkeypatch.setattr(module, "_objective_context", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        if operation == "extremum":
            s.certify_box_extremum(source, x, [(-1, 1)], [0], budget=budget)
        else:
            s.optimize_box_certified(source, x, [(-1, 1)], budget=budget)
