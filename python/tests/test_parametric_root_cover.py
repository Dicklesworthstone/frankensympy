"""A root family over a parameter range requires every cell, not samples."""

from fractions import Fraction
import importlib
from itertools import product

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def rational(value):
    return Fraction(value.p, value.q)


def identities(variables):
    return tuple(variable.identity_hex for variable in variables)


def assert_cover(receipt, unknown_count, parameter_count):
    assert type(receipt) is s.ParametricRootCoverReceipt
    assert type(receipt.leaves) is tuple
    assert receipt.leaf_count == len(receipt.leaves)
    assert 1 <= receipt.leaf_count <= 1024
    assert len(receipt.residuals) == len(receipt.unknown_variables) == unknown_count
    assert len(receipt.parameters) == parameter_count
    assert receipt.domain == receipt.unknown_domain + receipt.parameter_domain
    assert identities(receipt.variables) == identities(receipt.unknown_variables + receipt.parameters)
    assert "every parameter" in receipt.claim and "fixed" in receipt.claim
    leaves = receipt.leaves
    for leaf in leaves:
        assert type(leaf) is s.ParametricRootBoxReceipt
        assert leaf.residuals == receipt.residuals
        assert identities(leaf.variables) == identities(receipt.variables)
        assert leaf.unknown_domain == receipt.unknown_domain
        assert leaf.precision_digits == receipt.precision_digits
        assert 0 <= leaf.contraction_bound < 1
        for (outer_lo, outer_hi), (lo, hi) in zip(receipt.parameter_domain, leaf.parameter_domain):
            assert outer_lo <= lo <= hi <= outer_hi
        for (outer_lo, outer_hi), (lo, hi) in zip(receipt.unknown_domain, leaf.root_enclosure):
            assert outer_lo < lo <= hi < outer_hi
    assert receipt.root_enclosure == tuple(
        (min(leaf.root_enclosure[i][0] for leaf in leaves),
         max(leaf.root_enclosure[i][1] for leaf in leaves))
        for i in range(unknown_count)
    )


def assert_interval_partition(receipt):
    cells = sorted(leaf.parameter_domain[0] for leaf in receipt.leaves)
    assert cells[0][0] == receipt.parameter_domain[0][0]
    assert cells[-1][1] == receipt.parameter_domain[0][1]
    assert all(left[1] == right[0] for left, right in zip(cells, cells[1:]))


def test_adaptive_cover_succeeds_where_one_uniform_box_refuses_without_numerical_seeds(monkeypatch):
    x, p = s.symbols("parameter_cover_x parameter_cover_p")
    source = (1+p*p)*x-1
    program = s.compile_residual_system([source], (x, p))
    with pytest.raises(ValueError):
        program.certify_parametric_root_box([(0, 2)], [(0, 2)], precision=16)
    numerical = importlib.import_module("sympy.solvers.numerical")

    def no_numerical_seed(*args, **kwargs):
        raise AssertionError("parameter coverage must not depend on a floating-point root seed")

    monkeypatch.setattr(numerical, "_iterate_system", no_numerical_seed)
    result = program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], precision=16)
    assert_cover(result, 1, 1)
    assert_interval_partition(result)
    assert result.leaf_count > 1
    assert result.residuals == (source,)
    assert result.unknown_domain == ((0, 2),)
    assert result.parameter_domain == ((0, 2),)
    assert result.precision_digits == 16
    assert identities(result.unknown_variables) == (_symbol_binding(x).identity_hex,)
    assert identities(result.parameters) == (_symbol_binding(p).identity_hex,)
    for leaf in result.leaves:
        lo, hi = map(rational, leaf.root_enclosure[0])
        for parameter in map(rational, leaf.parameter_domain[0]):
            assert lo <= 1/(1+parameter*parameter) <= hi


def test_nonlinear_unknown_root_family_retains_the_entire_parameter_interval():
    x, p = s.symbols("parameter_cover_sqrt_x parameter_cover_sqrt_p")
    result = s.compile_residual_system([x*x-p], (x, p)).cover_parametric_roots_certified(
        [(1, 2)], [(2, 3)], max_leaves=1,
    )
    assert_cover(result, 1, 1)
    assert result.leaf_count == 1
    lo, hi = map(rational, result.root_enclosure[0])
    assert lo > 0 and lo*lo <= 2 and hi*hi >= 3


def test_transcendental_parameter_cover_retains_both_extreme_root_values():
    x, p = s.symbols("parameter_cover_exp_x parameter_cover_exp_p")
    result = s.compile_residual_system([s.exp(p)*x-1], (x, p)).cover_parametric_roots_certified(
        [(0, 2)], [(0, 1)], precision=16,
    )
    assert_cover(result, 1, 1)
    assert_interval_partition(result)
    assert result.leaf_count > 1
    assert result.root_enclosure[0][0] < s.Rational(3, 8)
    assert result.root_enclosure[0][1] >= 1


def test_nested_argument_order_and_matrix_snapshot_preserve_unknown_and_parameter_roles():
    x, y, p, q = s.symbols("parameter_cover_cx parameter_cover_cy parameter_cover_cp parameter_cover_cq")
    residuals = s.Matrix([x+2*y-p, y-q])
    program = s.compile_residual_system(residuals, [(x, y), (p, q)])
    residuals[0, 0] = x+2*y-999
    result = program.cover_parametric_roots_certified([(-4, 4), (-2, 2)], [(-1, 1), (-1, 1)])
    assert_cover(result, 2, 2)
    assert result.residuals == (x+2*y-p, y-q)
    assert result.root_enclosure == ((-3, 3), (-1, 1))
    assert result.leaf_count == 1
    assert result.leaves[0].preconditioner == ((1, -2), (0, 1))
    assert identities(result.unknown_variables) == tuple(_symbol_binding(v).identity_hex for v in (x, y))
    assert identities(result.parameters) == tuple(_symbol_binding(v).identity_hex for v in (p, q))


def test_multiple_varying_parameter_axes_and_fixed_coordinates_are_all_covered():
    x, fixed, p, q = s.symbols("parameter_cover_grid_x parameter_cover_fixed parameter_cover_grid_p parameter_cover_grid_q")
    result = s.compile_residual_system([(1+p*p+q*q)*x-1], (x, fixed, p, q)).cover_parametric_roots_certified(
        [(0, 2)], [(7, 7), (0, 1), (0, 1)], precision=16,
    )
    assert_cover(result, 1, 3)
    assert result.leaf_count > 1
    area = Fraction(0)
    cells = []
    for leaf in result.leaves:
        assert leaf.parameter_domain[0] == (7, 7)
        p_bounds, q_bounds = [tuple(map(rational, pair)) for pair in leaf.parameter_domain[1:]]
        cells.append((p_bounds, q_bounds))
        area += (p_bounds[1]-p_bounds[0]) * (q_bounds[1]-q_bounds[0])
        lo, hi = map(rational, leaf.root_enclosure[0])
        for a, b in product(p_bounds, q_bounds):
            assert lo <= 1/(1+a*a+b*b) <= hi
    assert area == 1
    for i, first in enumerate(cells):
        for second in cells[i+1:]:
            assert any(a_hi <= b_lo or b_hi <= a_lo
                       for (a_lo, a_hi), (b_lo, b_hi) in zip(first, second))


def test_point_parameters_are_allowed_and_do_not_create_duplicate_cells():
    x, p, q = s.symbols("parameter_cover_point_x parameter_cover_point_p parameter_cover_point_q")
    result = s.compile_residual_system([x-p-q], (x, p, q)).cover_parametric_roots_certified(
        [(0, 4)], [(1, 1), (1, 1)],
    )
    assert_cover(result, 1, 2)
    assert result.leaf_count == 1
    assert result.root_enclosure == ((2, 2),)


def test_exact_tiny_unknown_and_parameter_boxes_never_pass_through_binary64():
    x, p = s.symbols("parameter_cover_tiny_x parameter_cover_tiny_p")
    tiny = Fraction(1, 10**400)
    result = s.compile_residual_system([x-p], (x, p)).cover_parametric_roots_certified(
        [(0, 4*tiny)], [(tiny, 3*tiny)], max_leaves=1,
    )
    exact = s.Rational(tiny.numerator, tiny.denominator)
    assert_cover(result, 1, 1)
    assert result.root_enclosure == ((exact, 3*exact),)


def test_maximum_total_dimension_uses_fifteen_unknowns_and_one_parameter():
    unknowns = tuple(s.Symbol(f"parameter_cover_dimension_{i}") for i in range(15))
    p = s.Symbol("parameter_cover_dimension_p")
    result = s.compile_residual_system([x-p for x in unknowns], unknowns+(p,)).cover_parametric_roots_certified(
        [(-2, 2)]*15, [(-1, 1)], max_leaves=1,
    )
    assert_cover(result, 15, 1)
    assert result.root_enclosure == ((-1, 1),)*15


def test_original_parameter_dependency_is_resolved_by_refinement_without_algebraic_erasure():
    x, p = s.symbols("parameter_cover_dependency_x parameter_cover_dependency_p")
    denominator = s.Add(1, s.Mul(p, p, evaluate=False), evaluate=False)
    source = s.Add(x, s.Mul(-1, s.Pow(denominator, -1, evaluate=False), evaluate=False), evaluate=False)
    result = s.compile_residual_system([source], (x, p)).cover_parametric_roots_certified(
        [(0, 2)], [(-1, 1)], precision=16,
    )
    assert_cover(result, 1, 1)
    assert_interval_partition(result)
    # The midpoint/radius product on [-1,0] gives [-1/2,1], so its
    # reciprocal [1/2,2] touches X's boundary and needs a second split.
    # The four quarter cells have image hull [1/2,8/7].
    assert result.leaf_count == 4
    assert result.root_enclosure == ((s.Rational(1, 2), s.Rational(8, 7)),)


def test_missing_roots_at_some_parameter_values_cannot_publish_an_incomplete_family():
    x, p = s.symbols("parameter_cover_missing_x parameter_cover_missing_p")
    program = s.compile_residual_system([x-p*p], (x, p))
    # The first quarter certifies, but at p=1 its root x=1 leaves X.
    local = program.certify_parametric_root_box(
        [(Fraction(-1, 4), Fraction(1, 4))], [(0, Fraction(1, 4))],
    )
    assert local.root_enclosure[0][0] <= 0 <= local.root_enclosure[0][1]
    with pytest.raises(ValueError, match="inconclusive"):
        program.cover_parametric_roots_certified(
            [(Fraction(-1, 4), Fraction(1, 4))], [(0, 1)], max_leaves=16,
        )


@pytest.mark.parametrize("case", ["singular", "parameter_endpoint", "unknown_boundary", "fixed_singular"])
def test_singular_or_boundary_families_never_become_complete_by_subdivision(case):
    x, p = s.symbols("parameter_cover_refusal_x parameter_cover_refusal_p")
    source, unknown_box, parameter_box = {
        "singular": (x*x-p, [(-1, 1)], [(0, 1)]),
        "parameter_endpoint": (p*x-1, [(0, 2)], [(0, 1)]),
        "unknown_boundary": (x-p, [(-1, 1)], [(-1, 1)]),
        "fixed_singular": (x*x-p, [(-1, 1)], [(0, 0)]),
    }[case]
    with pytest.raises(ValueError, match="inconclusive"):
        s.compile_residual_system([source], (x, p)).cover_parametric_roots_certified(
            unknown_box, parameter_box, max_leaves=8,
        )


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_held_parameter_endpoint_poles_are_checked_even_after_other_cells_succeed(hidden):
    x, p = s.symbols("parameter_cover_pole_x parameter_cover_pole_p")
    inverse = s.Pow(s.Add(p, -1, evaluate=False), -1, evaluate=False)
    if hidden == "zero_product":
        hidden_term = s.Mul(0, inverse, evaluate=False)
    else:
        hidden_term = s.Add(s.Pow(inverse, 0, evaluate=False), -1, evaluate=False)
    source = s.Add(x, hidden_term, evaluate=False)
    with pytest.raises(ValueError):
        s.compile_residual_system([source], (x, p)).cover_parametric_roots_certified(
            [(-1, 1)], [(0, 1)], max_leaves=8,
        )


def test_a_leaf_limit_failure_preserves_consumption_without_exposing_partial_leaves():
    x, p = s.symbols("parameter_cover_limit_x parameter_cover_limit_p")
    program = s.compile_residual_system([(1+p*p)*x-1], (x, p))
    budget = ample_budget(max_memory_bytes=100*1024*1024)
    with pytest.raises(ValueError, match="inconclusive"):
        program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], max_leaves=1, budget=budget)
    assert budget.remaining_steps < 1 << 45
    assert budget.remaining_memory_bytes < 100*1024*1024 - 64*1024*1024
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], budget=budget)


def test_captured_assumptions_and_same_name_typed_roles_survive_later_mutation():
    x = s.Symbol("parameter_cover_positive_x", positive=True)
    p = s.Symbol("parameter_cover_positive_p", positive=True)
    original = [dict(v._assumptions) for v in (x, p)]
    expected = tuple(_symbol_binding(v).identity_hex for v in (x, p))
    program = s.compile_residual_system([x-p], (x, p))
    try:
        for variable in (x, p):
            variable._assumptions.clear()
            variable._assumptions["negative"] = True
        result = program.cover_parametric_roots_certified([(Fraction(1, 2), 3)], [(1, 2)])
        assert identities(result.variables) == expected
        with pytest.raises(ValueError, match="positive"):
            program.cover_parametric_roots_certified([(-1, 3)], [(1, 2)])
        with pytest.raises(ValueError, match="positive"):
            program.cover_parametric_roots_certified([(Fraction(1, 2), 3)], [(-1, 2)])
    finally:
        for variable, assumptions in zip((x, p), original):
            variable._assumptions.clear()
            variable._assumptions.update(assumptions)
    x, p = s.Dummy("same"), s.Dummy("same")
    result = s.compile_residual_system([x-p], (x, p)).cover_parametric_roots_certified(
        [(-2, 2)], [(-1, 1)],
    )
    assert result.unknown_variables[0].identity_hex != result.parameters[0].identity_hex
    assert result.root_enclosure == ((-1, 1),)


@pytest.mark.parametrize("value", [True, 0.0, "0", s.Float(0.0)])
def test_both_domain_groups_require_explicit_exact_bounds(value):
    x, p = s.symbols("parameter_cover_exact_x parameter_cover_exact_p")
    program = s.compile_residual_system([x-p], (x, p))
    with pytest.raises(TypeError, match="exact"):
        program.cover_parametric_roots_certified([(value, 2)], [(-1, 1)])
    with pytest.raises(TypeError, match="exact"):
        program.cover_parametric_roots_certified([(-2, 2)], [(value, 1)])


def test_controls_shapes_source_literals_and_scalar_height_are_preflighted():
    x, p = s.symbols("parameter_cover_invalid_x parameter_cover_invalid_p")
    program = s.compile_residual_system([x-p], (x, p))
    for precision in (7, 129, True, 16.0):
        with pytest.raises(ValueError, match="precision"):
            program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], precision=precision)
    for max_leaves in (0, 1025, True, 2.0):
        with pytest.raises(ValueError, match="max_leaves"):
            program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], max_leaves=max_leaves)
    for unknown_box, parameter_box in [([], [(-1, 1)]), ([(-2, 2)], [])]:
        with pytest.raises(TypeError, match="entries"):
            program.cover_parametric_roots_certified(unknown_box, parameter_box)
    with pytest.raises(ValueError, match="positive width"):
        program.cover_parametric_roots_certified([(0, 0)], [(-1, 1)])
    with pytest.raises(ValueError, match="lower bound"):
        program.cover_parametric_roots_certified([(-2, 2)], [(1, -1)])
    with pytest.raises(ValueError, match="4096-bit"):
        program.cover_parametric_roots_certified([(-2, 2)], [(Fraction(1, 1 << 4096), 1)])
    with pytest.raises(ValueError):
        s.compile_residual_system([x], x).cover_parametric_roots_certified([(-1, 1)], [])
    source = s.Add(x, -p, s.Float(0.5), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        s.compile_residual_system([source], (x, p)).cover_parametric_roots_certified([(-2, 2)], [(-1, 1)])


def test_strict_admission_does_not_invoke_user_numeric_or_container_hooks():
    x, p = s.symbols("parameter_cover_hook_x parameter_cover_hook_p")
    program = s.compile_residual_system([x-p], (x, p))
    calls = []

    class Numeric:
        def __float__(self):
            calls.append("float")
            return 0.0

        def __int__(self):
            calls.append("int")
            return 0

    class CustomList(list):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

    class CustomInt(int):
        def __int__(self):
            calls.append("custom_int")
            return super().__int__()

    for value in (Numeric(), CustomInt(0)):
        with pytest.raises(TypeError):
            program.cover_parametric_roots_certified([(-2, 2)], [(value, 1)])
    with pytest.raises(TypeError):
        program.cover_parametric_roots_certified(CustomList([(-2, 2)]), [(-1, 1)])
    with pytest.raises(TypeError):
        program.cover_parametric_roots_certified([(-2, 2)], CustomList([(-1, 1)]))
    with pytest.raises(TypeError):
        program.cover_parametric_roots_certified([(-2, 2)], [CustomList([-1, 1])])
    assert calls == []


def test_complete_receipt_and_each_local_receipt_are_owned_and_immutable():
    x, p = s.symbols("parameter_cover_receipt_x parameter_cover_receipt_p")
    unknown_box, parameter_box = [[0, 2]], [[0, 2]]
    result = s.compile_residual_system([(1+p*p)*x-1], (x, p)).cover_parametric_roots_certified(
        unknown_box, parameter_box, precision=16,
    )
    unknown_box[0][:] = [100, 200]
    parameter_box[0][:] = [100, 200]
    assert result.unknown_domain == ((0, 2),)
    assert result.parameter_domain == ((0, 2),)
    leaf = result.leaves[0]
    for receipt in (result, leaf):
        with pytest.raises(AttributeError):
            receipt.unknown_domain = ((100, 200),)
        with pytest.raises(AttributeError):
            del receipt._receipt
        with pytest.raises(AttributeError):
            receipt.__init__(receipt._receipt)
    with pytest.raises(TypeError):
        s.ParametricRootCoverReceipt(object())
    with pytest.raises(TypeError):
        _native.ParametricRootCoverReceipt()
    with pytest.raises(AttributeError):
        result._receipt.leaf_count = 0
    from sympy.utilities import ParametricRootCoverReceipt
    assert ParametricRootCoverReceipt is s.ParametricRootCoverReceipt


def test_native_search_preserves_exact_domains_and_returns_only_verified_covers():
    x, p = s.symbols("parameter_cover_native_x parameter_cover_native_p")
    native = s.compile_residual_system([(1+p*p)*x-1], (x, p)).native_system
    box = [(_native_expr(0), _native_expr(2))]
    with pytest.raises(TypeError):
        native.cover_parametric_roots(box, [(0.0, 2.0)])
    with pytest.raises(ValueError, match="inconclusive"):
        native.cover_parametric_roots(box, box, 16, 1)
    result = native.cover_parametric_roots(box, box, 16, 64)
    assert type(result) is _native.ParametricRootCoverReceipt
    assert result.leaf_count == len(result.leaves) > 1
    assert len(result.unknown_variables) == len(result.parameters) == 1
    assert result.precision_digits == 16


def test_supplied_budget_covers_both_native_phases_and_is_not_refilled_between_calls():
    x, p = s.symbols("parameter_cover_budget_x parameter_cover_budget_p")
    program = s.compile_residual_system([(1+p*p)*x-1], (x, p))
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.cover_parametric_roots_certified(
            [(0, 2)], [(0, 2)], precision=16, budget=ample_budget(max_memory_bytes=100*1024*1024),
        )
    probe = ample_budget()
    result = program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], precision=16, budget=probe)
    assert result.leaf_count > 1
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 128*1024*1024
    exact = ample_budget(max_memory_bytes=cost)
    program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], precision=16, budget=exact)
    assert exact.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.cover_parametric_roots_certified([(0, 2)], [(0, 2)], precision=16, budget=exact)


def test_one_private_fallback_is_created_before_preparation_and_forwarded_to_native(monkeypatch):
    x, p = s.symbols("parameter_cover_fallback_x parameter_cover_fallback_p")
    module = importlib.import_module("sympy.utilities._parametric_cover")
    original = module._call_budgeted
    created, observed = [], []

    def fallback():
        budget = ample_budget()
        created.append(budget)
        return budget

    def observe(function, *args, budget):
        observed.append((function.__name__, budget, budget.remaining_steps))
        return original(function, *args, budget=budget)

    monkeypatch.setattr(module, "_fallback_budget", fallback)
    monkeypatch.setattr(module, "_call_budgeted", observe)
    program = s.compile_residual_system([x-p], (x, p))
    program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)])
    assert len(created) == len(observed) == 1
    assert observed[0][0] == "cover_parametric_roots"
    assert observed[0][1] is created[0]
    assert observed[0][2] < 1 << 45
    supplied = ample_budget()
    observed.clear()
    program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], budget=supplied)
    assert len(created) == len(observed) == 1
    assert observed[0][1] is supplied


def test_exhaustion_and_final_publication_cancellation_keep_their_operational_types(monkeypatch):
    x, p = s.symbols("parameter_cover_cancel_x parameter_cover_cancel_p")
    program = s.compile_residual_system([x-p], (x, p))
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], budget=cancelled)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], budget=ample_budget(max_steps=0))
    module = importlib.import_module("sympy.utilities._parametric_cover")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_complete_acceptance(function, *args, **kwargs):
        result = original(function, *args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_complete_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        program.cover_parametric_roots_certified([(-2, 2)], [(-1, 1)], budget=budget)


def test_an_unchecked_native_result_cannot_authorize_complete_parameter_coverage(monkeypatch):
    x, p = s.symbols("parameter_cover_forged_x parameter_cover_forged_p")
    module = importlib.import_module("sympy.utilities._parametric_cover")
    monkeypatch.setattr(module, "_call_budgeted", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="accepted native"):
        s.compile_residual_system([x-p], (x, p)).cover_parametric_roots_certified(
            [(-2, 2)], [(-1, 1)],
        )
