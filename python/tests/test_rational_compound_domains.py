"""Compositional exact real calculus on bounded set expressions.

Images and extrema use literal exact sets, independently derived polynomial
values, and exact indexed-root comparisons. Held-source cases retain the
project's original-domain contract; no approximate sample proves coverage.
"""

import importlib
from unittest.mock import patch

import pytest
import sympy as s


def test_continuity_domain_is_an_input_to_range_extrema_and_stationary_points():
    x = s.Symbol("compound_continuity_x")
    domain = s.continuous_domain(1/x, x, s.S.Reals)
    expected = s.Union(s.Interval.open(-s.oo, 0), s.Interval.open(0, s.oo))
    assert domain == expected
    assert s.function_range(1/x, x, domain) == expected
    assert s.imageset(x, 1/x, domain) == expected
    assert s.minimum(1/x, x, domain) == -s.oo
    assert s.maximum(1/x, x, domain) == s.oo
    assert s.stationary_points(1/x, x, domain) == s.S.EmptySet


@pytest.mark.parametrize("included,expected", [
    (False, lambda: s.Union(s.Interval.open(1, 4), s.FiniteSet(9))),
    (True, lambda: s.Union(s.Interval.Ropen(1, 4), s.FiniteSet(9))),
])
def test_isolated_points_control_attainment_without_filling_image_gaps(included, expected):
    x = s.Symbol("compound_attainment_x")
    parts = [s.Interval.open(-2, -1), s.Interval.open(1, 2), s.FiniteSet(3)]
    if included:
        parts.append(s.FiniteSet(1))
    domain = s.Union(*parts)
    assert s.function_range(x*x, x, domain) == expected()
    assert s.imageset(x, x*x, domain) == expected()
    assert s.minimum(x*x, x, domain) == 1
    assert s.maximum(x*x, x, domain) == 9


def test_finite_domain_filters_original_poles_before_evaluating_values():
    x = s.Symbol("compound_finite_x")
    domain = s.FiniteSet(-2, 0, 2)
    assert s.function_range(1/x, x, domain) == s.FiniteSet(s.Rational(-1, 2), s.Rational(1, 2))
    assert s.minimum(1/x, x, domain) == s.Rational(-1, 2)
    assert s.maximum(1/x, x, domain) == s.Rational(1, 2)
    assert s.stationary_points(7, x, domain) == domain
    assert s.stationary_points(x**3/3-x, x, s.FiniteSet(-2, -1, 0, 1, 2)) == s.FiniteSet(-1, 1)


@pytest.mark.parametrize("open_hole", [False, True])
def test_relative_interval_complements_preserve_boundary_inclusion(open_hole):
    x = s.Symbol("compound_complement_x")
    hole = s.Interval.open(-1, 1) if open_hole else s.Interval(-1, 1)
    domain = s.Complement(s.Interval(-2, 2), hole)
    expected = s.Interval(1, 4) if open_hole else s.Interval.Lopen(1, 4)
    assert s.function_range(x*x, x, domain) == expected
    assert s.minimum(x*x, x, domain) == 1
    assert s.maximum(x*x, x, domain) == 4


def test_intersection_and_complement_do_not_reintroduce_excluded_fibers():
    x = s.Symbol("compound_intersection_x")
    base = s.Union(s.Interval(-3, -1), s.Interval(1, 3))
    domain = s.Intersection(base, s.Complement(s.S.Reals, s.FiniteSet(-2, 2)))
    expected = s.Union(s.Interval.Ropen(1, 4), s.Interval.Lopen(4, 9))
    assert s.function_range(x*x, x, domain) == expected
    assert s.imageset(x, x*x, domain) == expected
    # Restoring either preimage attains 4 and joins the two image components.
    assert s.function_range(x*x, x, s.Union(domain, s.FiniteSet(2))) == s.Interval(1, 9)


def test_unbounded_compound_domain_retains_finite_unattained_extrema():
    x = s.Symbol("compound_unbounded_x")
    domain = s.Complement(s.S.Reals, s.Interval(-1, 1))
    expected = s.Union(s.Interval.open(-1, 0), s.Interval.open(0, 1))
    assert s.function_range(1/x, x, domain) == expected
    assert s.minimum(1/x, x, domain) == -1
    assert s.maximum(1/x, x, domain) == 1
    assert s.function_range(x*x, x, domain) == s.Interval.open(1, s.oo)


@pytest.mark.parametrize("kind", ["empty", "poles"])
def test_empty_effective_domains_publish_no_values_and_require_no_derivative(kind):
    x = s.Symbol("compound_empty_x")
    domain = (s.Intersection(s.FiniteSet(0, 1), s.Interval(2, 3))
              if kind == "empty" else s.FiniteSet(0))
    core = importlib.import_module("sympy.core")
    with patch.object(core, "diff_with_receipt", side_effect=AssertionError("empty domain differentiated")):
        assert s.function_range(1/x, x, domain) == s.S.EmptySet
        assert s.stationary_points(1/x, x, domain) == s.S.EmptySet
        for operation in (s.minimum, s.maximum):
            with pytest.raises(ValueError, match="empty domain"):
                operation(1/x, x, domain)


def test_equal_indexed_roots_with_different_bound_names_share_one_hole():
    x, t, u = s.symbols("compound_algebraic_x compound_bound_t compound_bound_u")
    first = s.CRootOf(t*t-2, t, 1)
    alias = s.CRootOf(u*u-2, u, 1)
    domain = s.Complement(s.Interval(0, 2), s.FiniteSet(first, alias))
    expected = s.Union(s.Interval.Ropen(0, 2), s.Interval.Lopen(2, 4))
    assert s.function_range(x*x, x, domain) == expected
    assert s.imageset(x, x*x, domain) == expected
    # A separate isolated spelling of the open endpoint restores attainment.
    restored = s.Union(s.Interval.Lopen(first, 2), s.FiniteSet(alias))
    assert s.function_range(x*x, x, restored) == s.Interval(2, 4)
    stationary = s.stationary_points(x**3/3-2*x, x, restored)
    from sympy.polys.rootoftools import _algebraic_comparison
    assert len(stationary) == 1
    assert _algebraic_comparison(next(iter(stationary)), first) == 0


def test_compound_domains_never_round_nearby_rational_boundaries_together():
    x = s.Symbol("compound_nearby_x")
    epsilon = s.Rational(1, 10**40)
    middle = 1+epsilon
    domain = s.Complement(s.Interval(1, 1+2*epsilon), s.FiniteSet(middle))
    expected = s.Union(s.Interval.Ropen(1, middle), s.Interval.Lopen(middle, 1+2*epsilon))
    assert s.function_range(x, x, domain) == expected
    assert s.imageset(x, x, domain) == expected


def test_held_cancellation_and_isolated_points_cannot_restore_a_source_hole():
    x = s.Symbol("compound_held_x")
    source = s.Mul((x-1)**3, s.Pow(x-1, -1, evaluate=False), evaluate=False)
    domain = s.Union(s.Interval.open(0, 2), s.FiniteSet(1, 3))
    expected = s.Union(s.Interval.open(0, 1), s.FiniteSet(4))
    assert s.function_range(source, x, domain) == expected
    assert s.imageset(x, source, domain) == expected
    assert s.minimum(source, x, domain) == 0
    assert s.maximum(source, x, domain) == 4
    assert s.stationary_points(source, x, domain) == s.S.EmptySet
    hidden = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    assert s.function_range(hidden, x, s.FiniteSet(0, 1)) == s.FiniteSet(0)
    assert s.stationary_points(hidden, x, s.FiniteSet(0, 1)) == s.FiniteSet(1)


@pytest.mark.parametrize("factory", [s.Symbol, s.Dummy])
def test_compound_calculus_uses_exact_coordinate_identity(factory):
    x = factory("compound_typed", real=True)
    stranger = factory("compound_typed", positive=True)
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    assert s.function_range(x*x, x, domain) == s.Interval(1, 4)
    with pytest.raises(NotImplementedError):
        s.function_range(x*x, stranger, domain)


@pytest.mark.parametrize("elementary", [False, True])
def test_disconnected_range_uses_one_ledger_and_two_complete_native_charts(elementary):
    x = s.Symbol("compound_shared_work_x")
    domain = s.Union(*(s.Interval(3*i, 3*i+1) for i in range(8)))
    expression = s.exp(x) if elementary else x
    expected = (s.Union(*(s.Interval(s.exp(3*i), s.exp(3*i+1)) for i in range(8)))
                if elementary else domain)
    core = importlib.import_module("sympy.core")
    ranges = importlib.import_module("sympy.calculus._rational_range")
    original_work = ranges._Work
    meters = []

    class ObservedWork(original_work):
        def __init__(self):
            super().__init__()
            meters.append(self)

    original = core._native.solve_real_rational_inequality_native
    with patch.object(ranges, "_Work", ObservedWork), \
         patch.object(core._native, "solve_real_rational_inequality_native", wraps=original) as charts:
        assert s.function_range(expression, x, domain) == expected
    assert len(meters) == 1 and meters[0].operations > 0
    assert charts.call_count == 2


def test_elementary_range_composes_with_an_elementary_continuity_domain():
    x = s.Symbol("compound_elementary_continuity")
    expression = s.log((x-1)*(x+1))
    domain = s.continuous_domain(expression, x, s.S.Reals)
    assert domain == s.Union(s.Interval.open(-s.oo, -1), s.Interval.open(1, s.oo))
    assert s.function_range(expression, x, domain) == s.S.Reals
    assert s.minimum(expression, x, domain) == -s.oo
    assert s.maximum(expression, x, domain) == s.oo


@pytest.mark.parametrize("source,domain,expected,infimum,supremum", [
    (
        lambda x: s.exp(x),
        lambda: s.Complement(s.Interval(-2, 2), s.FiniteSet(0)),
        lambda: s.Union(s.Interval.Ropen(s.exp(-2), 1),
                        s.Interval.Lopen(1, s.exp(2))),
        lambda: s.exp(-2), lambda: s.exp(2),
    ),
    (
        lambda x: s.sqrt(x),
        lambda: s.Union(s.FiniteSet(-1, 0, 4), s.Interval.open(9, 16)),
        lambda: s.Union(s.FiniteSet(0, 2), s.Interval.open(3, 4)),
        lambda: 0, lambda: 4,
    ),
    (
        lambda x: s.cosh(x),
        lambda: s.Complement(s.Union(s.Interval(-3, -1), s.Interval(1, 3)),
                             s.FiniteSet(-2, 2)),
        lambda: s.Union(s.Interval.Ropen(s.cosh(1), s.cosh(2)),
                        s.Interval.Lopen(s.cosh(2), s.cosh(3))),
        lambda: s.cosh(1), lambda: s.cosh(3),
    ),
    (
        lambda x: s.log(x),
        lambda: s.FiniteSet(-1, 0, 1, 2),
        lambda: s.FiniteSet(0, s.log(2)),
        lambda: 0, lambda: s.log(2),
    ),
])
def test_elementary_images_keep_compound_domain_holes_and_points(
        source, domain, expected, infimum, supremum):
    x = s.Symbol("compound_elementary_image")
    expression, admitted = source(x), domain()
    assert s.function_range(expression, x, admitted) == expected()
    assert s.minimum(expression, x, admitted) == infimum()
    assert s.maximum(expression, x, admitted) == supremum()


def test_elementary_compound_domain_retains_a_held_rational_pole():
    x = s.Symbol("compound_elementary_original_pole")
    inner = s.Mul(x*x-1, s.Pow(x-1, -1, evaluate=False), evaluate=False)
    expression = s.exp(inner, evaluate=False)
    domain = s.Union(s.Interval(-1, 1), s.FiniteSet(2))
    assert s.function_range(expression, x, domain) == s.Union(
        s.Interval.Ropen(1, s.exp(2)), s.FiniteSet(s.exp(3)),
    )


def test_elementary_compound_domain_preserves_an_exact_algebraic_hole():
    x, z = s.symbols("compound_elementary_axis compound_elementary_bound")
    root = s.CRootOf(z*z-2, 1)
    domain = s.Complement(s.Interval(-2, 2), s.FiniteSet(root))
    assert s.function_range(s.exp(x), x, domain) == s.Union(
        s.Interval.Ropen(s.exp(-2), s.exp(root)),
        s.Interval.Lopen(s.exp(root), s.exp(2)),
    )


def test_elementary_compound_branch_can_be_empty():
    x = s.Symbol("compound_elementary_empty")
    domain = s.Union(s.Interval(-2, -1), s.FiniteSet(0))
    assert s.function_range(s.log(x), x, domain) == s.S.EmptySet
    for operation in (s.minimum, s.maximum):
        with pytest.raises(ValueError, match="empty domain"):
            operation(s.log(x), x, domain)


def test_elementary_range_rejects_changed_finite_domain_before_publication(monkeypatch):
    core = importlib.import_module("sympy.core")
    x = s.Symbol("compound_elementary_mutated_finite")
    domain = s.FiniteSet(0, 1)
    original = domain._surface_elements
    solve = core._native.solve_real_rational_inequality_native

    def changed(*args):
        result = solve(*args)
        domain._surface_elements = (s.Integer(2), s.Integer(3))
        return result

    monkeypatch.setattr(core._native, "solve_real_rational_inequality_native", changed)
    try:
        with pytest.raises(NotImplementedError, match="domain elements changed"):
            s.function_range(s.exp(x), x, domain)
    finally:
        domain._surface_elements = original


def test_elementary_range_refuses_an_oversized_domain_before_partial_solving(monkeypatch):
    core = importlib.import_module("sympy.core")
    x = s.Symbol("compound_elementary_oversized")
    domain = s.FiniteSet(*range(129))

    def forbidden(*args):
        raise AssertionError("an oversized elementary domain reached a sign chart")

    monkeypatch.setattr(core._native, "solve_real_rational_inequality_native", forbidden)
    with pytest.raises(NotImplementedError):
        s.function_range(s.exp(x), x, domain)


@pytest.mark.parametrize("operation", ["range", "minimum", "maximum", "stationary", "image"])
def test_compound_native_failure_is_terminal_for_every_consumer(operation):
    x = s.Symbol("compound_failed_x")
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    calls = {
        "range": lambda: s.function_range(x*x, x, domain),
        "minimum": lambda: s.minimum(x*x, x, domain),
        "maximum": lambda: s.maximum(x*x, x, domain),
        "stationary": lambda: s.stationary_points(x*x, x, domain),
        "image": lambda: s.imageset(x, x*x, domain),
    }
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=ValueError("compound native budget exhausted")) as charts:
        with pytest.raises(NotImplementedError, match="budget exhausted"):
            calls[operation]()
    assert charts.call_count == 1


def test_cancellation_after_source_admission_stops_before_derivative_work():
    x = s.Symbol("compound_cancelled_x")
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    core = importlib.import_module("sympy.core")
    ranges = importlib.import_module("sympy.calculus._rational_range")
    original = core._native.solve_real_rational_inequality_native
    cancelled = False

    def checked(*args):
        nonlocal cancelled
        result = original(*args)
        cancelled = True
        return result

    def checkpoint(_steps):
        if cancelled:
            raise RuntimeError("compound operation cancelled")

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=checked) as charts, \
         patch.object(ranges, "_charge_nested_if_active", side_effect=checkpoint), \
         patch.object(core, "diff_with_receipt", side_effect=AssertionError("cancelled work differentiated")):
        with pytest.raises(RuntimeError, match="cancelled"):
            s.function_range(x*x, x, domain)
    assert charts.call_count == 1


@pytest.mark.parametrize("target", ["domain", "source", "domain_arguments"])
def test_mutated_original_storage_is_rejected_before_range_publication(target):
    x = s.Symbol("compound_mutation_x")
    source = s.Add(x*x, 1, evaluate=False)
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    replacement = s.Interval(4, 5)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        if target == "domain":
            domain._native_set = replacement._native_set
        elif target == "source":
            source._args = (x, s.Integer(1))
        else:
            domain._py_args = (replacement,)
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="changed"):
            s.function_range(source, x, domain)


def test_modified_compound_set_accessor_is_never_invoked():
    x = s.Symbol("compound_modified_owner_x")
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))

    def forbidden(_):
        raise AssertionError("changed set accessor executed")

    with patch.object(s.Union, "args", property(forbidden)):
        with pytest.raises(NotImplementedError):
            s.function_range(x*x, x, domain)


@pytest.mark.parametrize("owner", [s.Union, s.Intersection, s.Complement])
def test_replaced_compound_storage_descriptor_is_rejected_without_callback(owner):
    x = s.Symbol("compound_changed_storage_descriptor_x")
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    observed = []

    def forged(_):
        observed.append("foreign storage accessor")
        return None

    core = importlib.import_module("sympy.core")
    with patch.object(owner, "_py_args", property(forged)), \
         patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("changed owner reached native solving")):
        with pytest.raises(NotImplementedError):
            s.function_range(x*x, x, domain)
    assert observed == []


def test_finite_domain_cannot_use_stale_native_members():
    x = s.Symbol("compound_finite_stale_x")
    domain = s.FiniteSet(1, 2)
    domain._surface_elements = (s.Integer(3), s.Integer(4))
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("stale finite members reached solving")):
        with pytest.raises(NotImplementedError, match="elements.*native storage"):
            s.function_range(x*x, x, domain)


@pytest.mark.parametrize("finite_wrapper", [False, True])
def test_domain_wrapper_cannot_reinterpret_a_different_native_set_kind(finite_wrapper):
    x = s.Symbol("compound_forged_domain_owner_x")
    if finite_wrapper:
        domain = s.FiniteSet(42)
        domain._surface_elements = None
        domain._native_set = s.Interval(0, 1)._native_set
    else:
        domain = s.Interval(0, 1)
        domain._native_set = s.FiniteSet(42)._native_set
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("changed set kind reached solving")):
        with pytest.raises(NotImplementedError, match="class.*native storage"):
            s.function_range(x*x, x, domain)


@pytest.mark.parametrize("replace_tuple", [False, True])
def test_finite_domain_member_mutation_prevents_publication(replace_tuple):
    x, t = s.symbols("compound_finite_mutation_x compound_finite_root_t")
    root = s.CRootOf(t*t-2, t, 1)
    domain = s.FiniteSet(root)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        if replace_tuple:
            domain._surface_elements = (s.Integer(3),)
        else:
            root._args = (t*t-3, s.Integer(1))
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="changed"):
            s.function_range(x*x, x, domain)


def test_finite_domain_storage_descriptor_is_rejected_without_callback():
    x = s.Symbol("compound_finite_descriptor_x")
    domain = s.FiniteSet(1, 2)
    observed = []

    def forged(_):
        observed.append("foreign finite members")
        return (s.Integer(3),)

    with patch.object(s.FiniteSet, "_surface_elements", property(forged)):
        with pytest.raises(NotImplementedError):
            s.function_range(x*x, x, domain)
    assert observed == []


def test_unsupported_or_oversized_compound_domains_never_trigger_partial_solving():
    x, unknown = s.symbols("compound_unsupported_x compound_unknown_bound")
    core = importlib.import_module("sympy.core")
    cases = (
        s.Union(s.Interval(-2, -1), s.Interval(unknown, 2)),
        s.Union(s.Interval(-2, -1), s.FiniteSet(s.I)),
        s.Union(s.Interval(-2, -1), s.S.Integers),
    )
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("unsupported domain reached a partial solver")):
        for domain in cases:
            with pytest.raises(NotImplementedError):
                s.function_range(x*x, x, domain)
            assert isinstance(s.imageset(x, x*x, domain), s.ImageSet)
        oversized = s.FiniteSet(*range(129))
        with pytest.raises(NotImplementedError, match="finite points"):
            s.function_range(x*x, x, oversized)
