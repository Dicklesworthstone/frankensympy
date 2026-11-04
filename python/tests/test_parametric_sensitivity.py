"""Implicit derivative bounds must retain the entire accepted root family."""

from fractions import Fraction
import importlib

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def rational(value):
    return Fraction(value.p, value.q)


def identities(variables):
    return tuple(variable.identity_hex for variable in variables)


def affine_family(kind="box"):
    x, p = s.symbols("sensitivity_affine_x sensitivity_affine_p")
    program = s.compile_residual_system([x-p], (x, p))
    if kind == "cover":
        return program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)])
    return program.certify_parametric_root_box([(-2, 2)], [(-1, 1)])


def assert_binding(result, family):
    assert type(result) is s.ParametricSensitivityReceipt
    assert result.residuals == family.residuals
    assert identities(result.variables) == identities(family.variables)
    assert identities(result.unknown_variables) == identities(family.unknown_variables)
    assert identities(result.parameters) == identities(family.parameters)
    assert result.unknown_domain == family.unknown_domain
    assert result.parameter_domain == family.parameter_domain
    assert result.domain == result.unknown_domain + result.parameter_domain
    assert result.root_enclosure == family.root_enclosure
    assert result.precision_digits == family.precision_digits
    assert result.leaf_count == getattr(family, "leaf_count", 1)
    assert 0 <= result.contraction_bound < 1
    assert type(result.solution_jacobian) is tuple
    assert len(result.solution_jacobian) == len(result.unknown_variables)
    for row in result.solution_jacobian:
        assert type(row) is tuple
        assert len(row) == len(result.parameters)
        assert all(type(interval) is tuple and interval[0] <= interval[1] for interval in row)
    assert "every implicit derivative" in result.claim
    assert "complete declared parameter box" in result.claim


@pytest.mark.parametrize("kind", ["box", "cover"])
def test_coupled_affine_sensitivities_are_exact_and_keep_both_coordinate_orders(kind):
    x, y, p, q = s.symbols("sensitivity_x sensitivity_y sensitivity_p sensitivity_q")
    source = s.Matrix([x+2*y-p, y-q])
    program = s.compile_residual_system(source, [(x, y), (p, q)])
    source[0, 0] = x+2*y-999
    if kind == "cover":
        family = program.cover_parametric_roots_certified([(-4, 4), (-2, 2)], [(-1, 1)]*2)
    else:
        family = program.certify_parametric_root_box([(-4, 4), (-2, 2)], [(-1, 1)]*2)
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    assert result.residuals == (x+2*y-p, y-q)
    # x=p-2q and y=q, so this checks signs as well as matrix orientation.
    assert result.solution_jacobian == (((1, 1), (-2, -2)), ((0, 0), (1, 1)))
    assert result.contraction_bound == 0
    assert identities(result.unknown_variables) == tuple(_symbol_binding(v).identity_hex for v in (x, y))
    assert identities(result.parameters) == tuple(_symbol_binding(v).identity_hex for v in (p, q))


def test_square_root_family_encloses_both_analytic_derivative_extremes():
    x, p = s.symbols("sensitivity_sqrt_x sensitivity_sqrt_p")
    family = s.compile_residual_system([x*x-p], (x, p)).certify_parametric_root_box(
        [(1, 2)], [(2, 3)], precision=16,
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    lo, hi = map(rational, result.solution_jacobian[0][0])
    # dx/dp=1/(2*sqrt(p)): the extreme squared slopes are 1/12 and 1/8.
    assert lo > 0 and lo*lo <= Fraction(1, 12)
    assert hi*hi >= Fraction(1, 8)
    assert result.contraction_bound == s.Rational(1, 3)


def test_preconditioner_is_not_mistaken_for_the_exact_inverse_derivative():
    x, p = s.symbols("sensitivity_defect_x sensitivity_defect_p")
    family = s.compile_residual_system([x/2-p], (x, p)).certify_parametric_root_box(
        [(-4, 4)], [(Fraction(-1, 2), Fraction(1, 2))], center=[0], preconditioner=[[1]],
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    lo, hi = result.solution_jacobian[0][0]
    # C=1 is an admissible preconditioner. The actual derivative is 2,
    # outside the erroneous first-order-only remainder [1/2, 3/2].
    assert lo <= 2 <= hi
    assert hi > s.Rational(3, 2)
    assert result.contraction_bound == s.Rational(1, 2)


def test_original_parameter_variation_is_retained_instead_of_its_midpoint():
    x, p = s.symbols("sensitivity_variation_x sensitivity_variation_p")
    family = s.compile_residual_system([x-p*p], (x, p)).certify_parametric_root_box(
        [(-1, 2)], [(0, 1)],
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    assert result.solution_jacobian == (((0, 2),),)


def test_complete_cover_derivative_hull_includes_every_leaf_and_shared_face():
    x, p = s.symbols("sensitivity_cover_x sensitivity_cover_p")
    family = s.compile_residual_system([(1+p*p)*x-1], (x, p)).cover_parametric_roots_certified(
        [(0, 2)], [(0, 2)], precision=16,
    )
    assert family.leaf_count > 1
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    checked_leaves = [leaf.enclose_sensitivity() for leaf in family.leaves]
    assert result.solution_jacobian == (((
        min(leaf.solution_jacobian[0][0][0] for leaf in checked_leaves),
        max(leaf.solution_jacobian[0][0][1] for leaf in checked_leaves),
    ),),)
    assert result.contraction_bound == max(leaf.contraction_bound for leaf in checked_leaves)
    # The independent analytic formula includes p=0 and p=2, not merely
    # whichever leaf the adaptive search visited first or last.
    lo, hi = map(rational, result.solution_jacobian[0][0])
    for parameter in (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2)):
        derivative = -2*parameter/(1+parameter*parameter)**2
        assert lo <= derivative <= hi


def test_transcendental_sensitivity_keeps_a_proved_positive_sign():
    x, p = s.symbols("sensitivity_sine_x sensitivity_sine_p")
    family = s.compile_residual_system([x-s.sin(p)], (x, p)).certify_parametric_root_box(
        [(-2, 2)], [(-1, 1)], precision=16,
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    lo, hi = result.solution_jacobian[0][0]
    assert s.Rational(2, 5) < lo < s.Rational(3, 5)
    assert hi >= 1


def test_fixed_and_unused_parameters_remain_derivative_columns():
    x, p, unused, q = s.symbols("sensitivity_columns_x sensitivity_columns_p sensitivity_unused sensitivity_columns_q")
    family = s.compile_residual_system([x-p-10*q], (x, p, unused, q)).certify_parametric_root_box(
        [(-20, 20)], [(-1, 1), (7, 7), (-1, 1)],
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    assert result.solution_jacobian == (((1, 1), (0, 0), (10, 10)),)


def test_typed_same_name_coordinates_and_captured_assumptions_do_not_rebind():
    x, p = s.Dummy("same"), s.Dummy("same")
    family = s.compile_residual_system([x-3*p], (x, p)).certify_parametric_root_box(
        [(-4, 4)], [(-1, 1)],
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    assert result.solution_jacobian == (((3, 3),),)
    assert result.unknown_variables[0].identity_hex != result.parameters[0].identity_hex

    x = s.Symbol("sensitivity_positive_x", positive=True)
    p = s.Symbol("sensitivity_positive_p", positive=True)
    family = s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
        [(Fraction(1, 2), 3)], [(1, 2)],
    )
    expected = identities(family.variables)
    original = [dict(variable._assumptions) for variable in (x, p)]
    try:
        for variable in (x, p):
            variable._assumptions.clear()
            variable._assumptions["negative"] = True
        result = family.enclose_sensitivity()
        assert identities(result.variables) == expected
        assert result.solution_jacobian == (((1, 1),),)
    finally:
        for variable, assumptions in zip((x, p), original):
            variable._assumptions.clear()
            variable._assumptions.update(assumptions)


def test_exact_derivative_larger_than_binary64_never_rounds_to_infinity():
    x, p = s.symbols("sensitivity_huge_x sensitivity_huge_p")
    huge = 10**400
    tiny = Fraction(1, huge)
    family = s.compile_residual_system([s.Rational(1, huge)*x-p], (x, p)).certify_parametric_root_box(
        [(-2, 2)], [(-tiny, tiny)], center=[0], preconditioner=[[huge]],
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    assert result.solution_jacobian == (((huge, huge),),)


@pytest.mark.parametrize("unknown_count", [1, 15])
def test_maximum_coordinate_envelope_preserves_every_row_and_column(unknown_count):
    unknowns = tuple(s.Symbol(f"sensitivity_max_x_{i}") for i in range(unknown_count))
    parameters = tuple(s.Symbol(f"sensitivity_max_p_{j}") for j in range(16-unknown_count))
    rhs = sum((j+1)*p for j, p in enumerate(parameters))
    family = s.compile_residual_system([x-rhs for x in unknowns], unknowns+parameters).certify_parametric_root_box(
        [(-1, 1)]*len(unknowns), [(0, 0)]*len(parameters),
    )
    result = family.enclose_sensitivity()
    assert_binding(result, family)
    row = tuple((j+1, j+1) for j in range(len(parameters)))
    assert result.solution_jacobian == (row,)*len(unknowns)


@pytest.mark.parametrize("kind", ["box", "cover"])
def test_native_entry_points_preserve_the_same_accepted_type(kind):
    family = affine_family(kind)
    native = family._receipt.enclose_sensitivity()
    assert type(native) is _native.ParametricSensitivityReceipt
    result = s.ParametricSensitivityReceipt(native)
    assert_binding(result, family)
    assert result.solution_jacobian == (((1, 1),),)
    with pytest.raises(TypeError):
        family._receipt.enclose_sensitivity(budget=object())


def test_receipt_is_immutable_and_cannot_be_forged_from_an_unchecked_object():
    family = affine_family()
    result = family.enclose_sensitivity()
    assert "solution_jacobian" in repr(result)
    with pytest.raises(AttributeError):
        result.solution_jacobian = (((999, 999),),)
    with pytest.raises(AttributeError):
        del result._receipt
    with pytest.raises(AttributeError):
        result.__init__(result._receipt)
    with pytest.raises(TypeError):
        s.ParametricSensitivityReceipt(object())
    with pytest.raises(TypeError):
        s.ParametricSensitivityReceipt(family._receipt)
    with pytest.raises(TypeError):
        _native.ParametricSensitivityReceipt()
    with pytest.raises(AttributeError):
        result._receipt.solution_jacobian = []
    projected = result._receipt.solution_jacobian
    projected[0].clear()
    assert result.solution_jacobian == (((1, 1),),)
    from sympy.utilities import ParametricSensitivityReceipt
    assert ParametricSensitivityReceipt is s.ParametricSensitivityReceipt


def test_sensitivity_is_self_contained_and_does_not_recompile_or_search(monkeypatch):
    family = affine_family()
    numerical = importlib.import_module("sympy.solvers.numerical")

    def unavailable(*args, **kwargs):
        raise AssertionError("sensitivity must use the accepted source instead of a new root search")

    monkeypatch.setattr(s, "compile_residual_system", unavailable)
    monkeypatch.setattr(numerical, "_iterate_system", unavailable)
    assert family.enclose_sensitivity().solution_jacobian == (((1, 1),),)


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_each_resource_refusal_keeps_the_original_family_usable(dimension):
    family = affine_family()
    with pytest.raises(s.EvaluationBudgetExceeded):
        family.enclose_sensitivity(budget=ample_budget(**{dimension: 0}))
    assert family.root_enclosure == ((-1, 1),)
    assert family.enclose_sensitivity().solution_jacobian == (((1, 1),),)


def test_root_acceptance_and_sensitivity_can_share_one_cumulative_budget():
    x, p = s.symbols("sensitivity_budget_x sensitivity_budget_p")
    budget = ample_budget(max_memory_bytes=100*1024*1024)
    family = s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
        [(-2, 2)], [(-1, 1)], center=[0], preconditioner=[[1]], budget=budget,
    )
    remaining = budget.remaining_memory_bytes
    assert remaining < 100*1024*1024-64*1024*1024
    with pytest.raises(s.EvaluationBudgetExceeded):
        family.enclose_sensitivity(budget=budget)
    assert budget.remaining_memory_bytes <= remaining
    assert family.root_enclosure == ((-1, 1),)
    assert family.enclose_sensitivity(budget=ample_budget()).solution_jacobian == (((1, 1),),)


def test_repeated_sensitivity_calls_never_reset_or_refund_the_ledger():
    family = affine_family()
    budget = ample_budget(max_memory_bytes=100*1024*1024)
    family.enclose_sensitivity(budget=budget)
    remaining = budget.remaining_memory_bytes
    assert remaining < 100*1024*1024-64*1024*1024
    with pytest.raises(s.EvaluationBudgetExceeded):
        family.enclose_sensitivity(budget=budget)
    assert budget.remaining_memory_bytes <= remaining


def test_one_private_ledger_is_created_and_supplied_ledgers_are_preserved(monkeypatch):
    family = affine_family()
    module = importlib.import_module("sympy.utilities._parametric_sensitivity")
    original = module._call_budgeted
    created, observed = [], []

    def fallback():
        budget = ample_budget()
        created.append(budget)
        return budget

    def observe(function, *args, budget):
        observed.append(budget)
        return original(function, *args, budget=budget)

    monkeypatch.setattr(module, "_fallback_budget", fallback)
    monkeypatch.setattr(module, "_call_budgeted", observe)
    family.enclose_sensitivity()
    assert len(created) == 1 and observed == created
    supplied = ample_budget()
    family.enclose_sensitivity(budget=supplied)
    assert len(created) == 1 and observed[-1] is supplied


def test_cancellation_before_work_and_after_native_acceptance_prevents_publication(monkeypatch):
    family = affine_family()
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        family.enclose_sensitivity(budget=cancelled)
    module = importlib.import_module("sympy.utilities._parametric_sensitivity")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_acceptance(function, *args, **kwargs):
        result = original(function, *args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        family.enclose_sensitivity(budget=budget)
    assert family.root_enclosure == ((-1, 1),)


def test_unchecked_native_return_and_invalid_budget_are_rejected(monkeypatch):
    family = affine_family()
    with pytest.raises(TypeError):
        family.enclose_sensitivity(budget=object())
    module = importlib.import_module("sympy.utilities._parametric_sensitivity")
    with pytest.raises(TypeError, match="accepted native"):
        module.enclose_sensitivity(object())
    monkeypatch.setattr(module, "_call_budgeted", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="accepted native"):
        family.enclose_sensitivity()
