"""Every root in a finite domain requires an independently checked cover."""

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


def assert_complete(receipt, count, domain):
    assert type(receipt) is s.RootIsolationReceipt
    assert receipt.root_count == count == len(receipt.roots)
    assert receipt.is_root_free is (count == 0)
    assert receipt.domain == tuple(domain)
    assert type(receipt.roots) is tuple
    assert 1 <= receipt.leaf_count <= 1024
    assert 0 <= receipt.excluded_leaf_count <= receipt.leaf_count
    assert "all roots are covered" in receipt.claim
    assert "closed box" in receipt.claim
    for root in receipt.roots:
        assert type(root) is s.RootBoxReceipt
        assert root.residuals == receipt.residuals
        assert tuple(v.identity_hex for v in root.variables) == tuple(
            v.identity_hex for v in receipt.variables
        )
        assert 0 <= root.contraction_bound < 1
        for (outer_lo, outer_hi), (lo, hi), (root_lo, root_hi) in zip(
                receipt.domain, root.domain, root.root_enclosure):
            assert outer_lo <= lo < root_lo <= root_hi < hi <= outer_hi
    for i, first in enumerate(receipt.roots):
        for second in receipt.roots[i + 1:]:
            assert any(a_hi < b_lo or b_hi < a_lo
                       for (a_lo, a_hi), (b_lo, b_hi) in zip(first.domain, second.domain))


def test_all_irrational_roots_are_discovered_without_a_numerical_initial_guess(monkeypatch):
    x = s.Symbol("isolation_irrational_x")
    source = x*x - 2
    numerical = importlib.import_module("sympy.solvers.numerical")

    def no_numerical_seed(*args, **kwargs):
        raise AssertionError("complete isolation must not depend on an approximate root guess")

    monkeypatch.setattr(numerical, "_iterate_system", no_numerical_seed)
    result = s.compile_residual_system([source], x).isolate_roots_certified(
        [(-2, 2)], precision=16,
    )
    assert_complete(result, 2, [(-2, 2)])
    assert result.residuals == (source,)
    assert result.variables[0].identity_hex == _symbol_binding(x).identity_hex
    assert result.precision_digits == 16
    negative, positive = result.roots
    lo, hi = map(rational, negative.root_enclosure[0])
    assert lo < hi < 0 and hi*hi <= 2 <= lo*lo
    lo, hi = map(rational, positive.root_enclosure[0])
    assert 0 < lo < hi and lo*lo <= 2 <= hi*hi
    assert result.excluded_leaf_count > 0


def test_coupled_two_dimensional_cover_contains_all_four_distinct_solutions():
    x, y = s.symbols("isolation_coupled_x isolation_coupled_y")
    residuals = [x*x - 1, x*x + y*y - 2]
    result = s.compile_residual_system(residuals, (x, y)).isolate_roots_certified(
        [(-2, 2), (-2, 2)], precision=16,
    )
    assert_complete(result, 4, [(-2, 2), (-2, 2)])
    for a in (-1, 1):
        for b in (-1, 1):
            assert sum(lo_x <= a <= hi_x and lo_y <= b <= hi_y
                       for root in result.roots
                       for (lo_x, hi_x), (lo_y, hi_y) in [root.root_enclosure]) == 1


def test_transcendental_roots_are_counted_across_a_complete_finite_domain():
    x = s.Symbol("isolation_sine_x")
    result = s.compile_residual_system([s.sin(x)], x).isolate_roots_certified(
        [(-4, 4)], precision=16,
    )
    assert_complete(result, 3, [(-4, 4)])
    negative, zero, positive = result.roots
    # The independently known roots in [-4,4] are -pi, 0, pi.
    assert negative.root_enclosure[0][1] < 0
    assert zero.root_enclosure[0][0] <= 0 <= zero.root_enclosure[0][1]
    assert positive.root_enclosure[0][0] > 0


def test_roots_on_internal_dyadic_faces_are_not_omitted_or_duplicated():
    x = s.Symbol("isolation_seam_x")
    result = s.compile_residual_system([x**3-x], x).isolate_roots_certified(
        [(-2, 2)], precision=16,
    )
    assert_complete(result, 3, [(-2, 2)])
    for expected in (-1, 0, 1):
        assert sum(root.root_enclosure[0][0] <= expected <= root.root_enclosure[0][1]
                   for root in result.roots) == 1
    zero = next(root for root in result.roots
                if root.root_enclosure[0][0] <= 0 <= root.root_enclosure[0][1])
    assert zero.domain[0][0] < 0 < zero.domain[0][1]


def test_empty_root_set_is_an_accepted_complete_exclusion_claim():
    x = s.Symbol("isolation_empty_x")
    result = s.compile_residual_system([x*x+1], x).isolate_roots_certified([(-2, 2)])
    assert_complete(result, 0, [(-2, 2)])
    assert result.roots == ()
    assert result.leaf_count == result.excluded_leaf_count == 1
    assert result.precision_digits == 32


@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_supported_affine_dimensions_have_one_complete_root_box(dimension):
    variables = tuple(s.Symbol(f"isolation_affine_{dimension}_{i}") for i in range(dimension))
    result = s.compile_residual_system(list(variables), variables).isolate_roots_certified(
        [(-1, 1)] * dimension,
    )
    assert_complete(result, 1, [(-1, 1)] * dimension)
    assert result.leaf_count == 1
    assert result.roots[0].root_enclosure == ((0, 0),) * dimension


def test_nested_argument_layout_and_matrix_source_are_captured_before_search():
    x, y = s.symbols("isolation_snapshot_x isolation_snapshot_y")
    source = s.Matrix([x+y-3, x-y+1])
    program = s.compile_residual_system(source, [(x,), y])
    source[0, 0] = x+y-300
    result = program.isolate_roots_certified([(-1, 3), (-1, 3)])
    assert_complete(result, 1, [(-1, 3), (-1, 3)])
    assert result.residuals == (x+y-3, x-y+1)
    assert result.roots[0].root_enclosure == ((1, 1), (2, 2))


def test_exact_tiny_root_and_box_never_pass_through_binary64():
    x = s.Symbol("isolation_tiny_x")
    tiny = Fraction(1, 10**400)
    source = x - s.Rational(tiny.numerator, tiny.denominator)
    result = s.compile_residual_system([source], x).isolate_roots_certified(
        [(0, 2*tiny)], max_leaves=1,
    )
    exact = s.Rational(tiny.numerator, tiny.denominator)
    assert_complete(result, 1, [(0, 2*exact)])
    assert result.roots[0].root_enclosure == ((exact, exact),)
    assert result.leaf_count == 1


def test_subdivision_recovers_an_original_denominator_interval_dependency():
    x = s.Symbol("isolation_dependency_x")
    denominator = s.Add(1, s.Mul(x, x, evaluate=False), evaluate=False)
    source = s.Pow(denominator, -1, evaluate=False)
    result = s.compile_residual_system([source], x).isolate_roots_certified([(-1, 1)])
    assert_complete(result, 0, [(-1, 1)])
    assert result.residuals == (source,)
    assert result.leaf_count == result.excluded_leaf_count == 2


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_nonzero_first_residual_never_hides_a_later_original_domain_hole(hidden):
    x, y = s.symbols("isolation_domain_x isolation_domain_y")
    inverse = s.Pow(s.Add(y, 1, evaluate=False), -1, evaluate=False)
    invalid = (s.Mul(0, inverse, evaluate=False) if hidden == "zero_product"
               else s.Pow(inverse, 0, evaluate=False))
    program = s.compile_residual_system([1, invalid], (x, y))
    with pytest.raises(ValueError):
        program.isolate_roots_certified([(-1, 1), (-1, 1)], max_leaves=8)


@pytest.mark.parametrize("case", ["singular", "outer_boundary", "mixed_simple_and_singular"])
def test_unresolved_roots_never_publish_partial_completeness(case):
    x = s.Symbol("isolation_unresolved_x")
    source, domain = {
        "singular": (x*x, (-1, 1)),
        "outer_boundary": (x, (0, 1)),
        "mixed_simple_and_singular": (x*x*(x+1), (-2, 2)),
    }[case]
    budget = ample_budget()
    before = budget.remaining_steps
    with pytest.raises(ValueError, match="unresolved|subdivision|complete"):
        s.compile_residual_system([source], x).isolate_roots_certified(
            [domain], max_leaves=16, budget=budget,
        )
    assert budget.remaining_steps < before


def test_captured_assumptions_and_typed_symbol_identity_survive_later_mutation():
    x = s.Symbol("isolation_positive_x", positive=True)
    binding = _symbol_binding(x)
    facts = dict(x._assumptions)
    program = s.compile_residual_system([x-1], x)
    try:
        x._assumptions.clear()
        x._assumptions["negative"] = True
        result = program.isolate_roots_certified([(Fraction(1, 2), 2)])
        assert result.variables[0].identity_hex == binding.identity_hex
        assert result.roots[0].root_enclosure == ((1, 1),)
        with pytest.raises(ValueError, match="positive"):
            program.isolate_roots_certified([(-1, 2)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


@pytest.mark.parametrize("value", [True, 0.0, "0", s.Float(0.0)])
def test_box_bounds_require_closed_exact_values(value):
    x = s.Symbol("isolation_exact_x")
    program = s.compile_residual_system([x], x)
    with pytest.raises(TypeError, match="exact"):
        program.isolate_roots_certified([(value, 1)])


@pytest.mark.parametrize("max_leaves", [0, 1025, 10**100, True, 1.0])
def test_subdivision_limits_require_bounded_builtin_integers(max_leaves):
    x = s.Symbol("isolation_limit_x")
    with pytest.raises(ValueError, match="max_leaves"):
        s.compile_residual_system([x], x).isolate_roots_certified([(-1, 1)], max_leaves=max_leaves)


def test_shape_source_rational_height_and_precision_refusals_precede_search():
    x = s.Symbol("isolation_invalid_x")
    source = s.Add(x, s.Float(0.5), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        s.compile_residual_system([source], x).isolate_roots_certified([(-1, 1)])
    program = s.compile_residual_system([x], x)
    for precision in (7, 129, True, 16.0):
        with pytest.raises(ValueError, match="precision"):
            program.isolate_roots_certified([(-1, 1)], precision=precision)
    with pytest.raises(ValueError, match="4096-bit"):
        program.isolate_roots_certified([(Fraction(1, 1 << 4096), 1)])
    with pytest.raises(ValueError, match="positive width"):
        program.isolate_roots_certified([(0, 0)])
    with pytest.raises(ValueError, match="lower bound"):
        program.isolate_roots_certified([(1, -1)])
    with pytest.raises(TypeError, match="entries"):
        program.isolate_roots_certified([])
    rectangular = s.compile_residual_system([x, x*x], x)
    with pytest.raises(ValueError, match="square"):
        rectangular.isolate_roots_certified([(-1, 1)])
    variables = tuple(s.Symbol(f"isolation_too_many_{i}") for i in range(17))
    with pytest.raises(ValueError, match="1..16"):
        s.compile_residual_system(list(variables), variables).isolate_roots_certified([(-1, 1)]*17)


def test_exact_admission_never_invokes_custom_numeric_or_container_hooks():
    x = s.Symbol("isolation_callbacks_x")
    program = s.compile_residual_system([x], x)
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
            program.isolate_roots_certified([(value, 1)])
    with pytest.raises(TypeError):
        program.isolate_roots_certified(CustomList([(-1, 1)]))
    with pytest.raises(TypeError):
        program.isolate_roots_certified([CustomList([-1, 1])])
    with pytest.raises(ValueError):
        program.isolate_roots_certified([(-1, 1)], max_leaves=CustomInt(4))
    assert calls == []


def test_global_and_contained_local_receipts_are_immutable_owned_native_acceptances():
    x = s.Symbol("isolation_receipt_x")
    domain = [[-1, 1]]
    result = s.compile_residual_system([x], x).isolate_roots_certified(domain)
    domain[0][:] = [100, 101]
    assert result.domain == ((-1, 1),)
    assert result.roots[0].root_enclosure == ((0, 0),)
    for receipt in (result, result.roots[0]):
        with pytest.raises(AttributeError):
            receipt.domain = ((100, 101),)
        with pytest.raises(AttributeError):
            del receipt._receipt
        with pytest.raises(AttributeError):
            receipt.__init__(receipt._receipt)
    with pytest.raises(TypeError):
        s.RootIsolationReceipt(object())
    with pytest.raises(TypeError):
        _native.RootIsolationReceipt()
    with pytest.raises(AttributeError):
        result._receipt.root_count = 100
    with pytest.raises(AttributeError):
        result._receipt.roots = ()
    from sympy.utilities import RootIsolationReceipt
    assert RootIsolationReceipt is s.RootIsolationReceipt


def test_native_entry_accepts_exact_boxes_and_exposes_only_frozen_checked_roots():
    x = s.Symbol("isolation_native_x")
    native = s.compile_residual_system([x], x).native_system
    with pytest.raises(TypeError):
        native.isolate_roots([(-1.0, 1.0)])
    with pytest.raises(ValueError):
        native.isolate_roots([(_native_expr(x), _native_expr(1))])
    with pytest.raises(ValueError):
        native.isolate_roots([(_native_expr(0), _native_expr(0))])
    result = native.isolate_roots([(_native_expr(-1), _native_expr(1))], 16, 1)
    assert type(result) is _native.RootIsolationReceipt
    assert result.root_count == 1 and result.leaf_count == 1
    assert result.is_root_free is False
    assert type(result.roots[0]) is _native.RootBoxReceipt


def test_search_and_verification_share_one_cumulative_caller_budget():
    x = s.Symbol("isolation_budget_x")
    program = s.compile_residual_system([x], x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.isolate_roots_certified(
            [(-1, 1)], budget=ample_budget(max_memory_bytes=100*1024*1024),
        )
    probe = ample_budget()
    program.isolate_roots_certified([(-1, 1)], budget=probe)
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 128*1024*1024
    exact = ample_budget(max_memory_bytes=cost)
    program.isolate_roots_certified([(-1, 1)], budget=exact)
    assert exact.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.isolate_roots_certified([(-1, 1)], budget=exact)


def test_cancelled_or_exhausted_work_and_final_publication_cannot_return_success(monkeypatch):
    x = s.Symbol("isolation_cancel_x")
    program = s.compile_residual_system([x], x)
    budget = ample_budget()
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.isolate_roots_certified([(-1, 1)], budget=budget)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.isolate_roots_certified([(-1, 1)], budget=ample_budget(max_steps=0))
    module = importlib.import_module("sympy.utilities._root_isolation")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_native_acceptance(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_native_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        program.isolate_roots_certified([(-1, 1)], budget=budget)


def test_a_forged_native_result_cannot_become_a_complete_root_receipt(monkeypatch):
    x = s.Symbol("isolation_forged_x")
    module = importlib.import_module("sympy.utilities._root_isolation")
    monkeypatch.setattr(module, "_call_budgeted", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="accepted native"):
        s.compile_residual_system([x], x).isolate_roots_certified([(-1, 1)])
