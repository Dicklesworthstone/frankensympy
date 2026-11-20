"""Exact images, extrema and stationary points on compound real domains.

Expectations use exact set equality. Ordinary supported examples were checked
against SymPy 1.14.0; the final elementary-fold examples additionally exercise
the direct monotone-image contract where that oracle may refuse. Held-source
and mutation cases exercise the repository's original-domain contract. Neither
numerical samples nor an incomplete solver establish any result.
"""

import importlib
from unittest.mock import patch

import pytest
import sympy as s


@pytest.fixture
def x():
    return s.Symbol("compound_calculus_x")


def test_disjoint_component_images_retain_a_real_gap(x):
    domain = s.Union(s.Interval(-3, -2), s.Interval(0, 1))
    expected = s.Union(s.Interval(0, 1), s.Interval(4, 9))
    assert s.function_range(x*x, x, domain) == expected
    assert s.minimum(x*x, x, domain) == 0
    assert s.maximum(x*x, x, domain) == 9
    assert s.stationary_points(x*x, x, domain) == s.FiniteSet(0)


def test_overlapping_component_images_are_merged(x):
    domain = s.Union(s.Interval(-3, -1), s.Interval(2, 4))
    assert s.function_range(x*x, x, domain) == s.Interval(1, 16)


def test_mixed_interval_and_isolated_point_image(x):
    domain = s.Union(s.Interval.open(-1, 1), s.FiniteSet(2))
    expected = s.Union(s.Interval.Ropen(0, 1), s.FiniteSet(4))
    assert s.function_range(x*x, x, domain) == expected
    assert s.minimum(x*x, x, domain) == 0
    assert s.maximum(x*x, x, domain) == 4


@pytest.mark.parametrize("removed,expected", [
    (s.Interval(-1, 1), s.Interval.open(1, s.oo)),
    (s.Interval.open(-1, 1), s.Interval(1, s.oo)),
])
def test_complement_endpoints_control_extremum_attainment(x, removed, expected):
    domain = s.Complement(s.S.Reals, removed)
    assert s.function_range(x*x, x, domain) == expected
    # maximum/minimum are supremum/infimum APIs, including unattained bounds.
    assert s.minimum(x*x, x, domain) == 1
    assert s.maximum(x*x, x, domain) == s.oo
    assert s.stationary_points(x*x, x, domain) == s.S.EmptySet


def test_finite_holes_split_an_interval_image(x):
    domain = s.Complement(s.Interval(0, 3), s.FiniteSet(1, 2))
    expected = s.Union(s.Interval.Ropen(0, 1), s.Interval.open(1, 2),
                       s.Interval.Lopen(2, 3))
    assert s.function_range(x, x, domain) == expected


def test_nested_set_operations_use_all_preimages(x):
    base = s.Union(s.Interval(-3, -1), s.Interval(1, 3))
    both = s.Intersection(base, s.Complement(s.S.Reals, s.FiniteSet(-2, 2)))
    one = s.Intersection(base, s.Complement(s.S.Reals, s.FiniteSet(-2)))
    expected = s.Union(s.Interval.Ropen(1, 4), s.Interval.Lopen(4, 9))
    assert s.function_range(x*x, x, both) == expected
    # Excluding one preimage does not remove a value attained on the other.
    assert s.function_range(x*x, x, one) == s.Interval(1, 9)


def test_finite_images_and_extrema_need_no_derivative(x):
    domain = s.FiniteSet(-2, 0, 3)
    core = importlib.import_module("sympy.core")
    with patch.object(core, "diff_with_receipt",
                      side_effect=AssertionError("finite image requested a derivative")):
        assert s.function_range(x*x, x, domain) == s.FiniteSet(0, 4, 9)
        assert s.minimum(x*x, x, domain) == 0
        assert s.maximum(x*x, x, domain) == 9


def test_finite_poles_are_removed_before_point_evaluation(x):
    domain = s.FiniteSet(0, 1, 2)
    assert s.function_range(1/x, x, domain) == s.FiniteSet(s.Rational(1, 2), 1)
    assert s.minimum(1/x, x, domain) == s.Rational(1, 2)
    assert s.maximum(1/x, x, domain) == 1
    assert s.stationary_points(1/x, x, domain) == s.S.EmptySet


def test_finite_removable_hole_is_not_reintroduced(x):
    source = s.Mul(x*x - 1, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
    assert s.function_range(source, x, s.FiniteSet(0, 1, 2)) == s.FiniteSet(1, 3)
    assert s.function_range(source, x, s.FiniteSet(1)) == s.S.EmptySet


@pytest.mark.parametrize("zero_power", [False, True])
def test_held_constant_function_retains_finite_domain_holes(x, zero_power):
    reciprocal = s.Pow(x, -1, evaluate=False)
    source = (s.Pow(reciprocal, 0, evaluate=False) if zero_power
              else s.Mul(0, reciprocal, evaluate=False))
    assert s.function_range(source, x, s.FiniteSet(0, 1)) == s.FiniteSet(int(zero_power))
    assert s.stationary_points(source, x, s.FiniteSet(0, 1)) == s.FiniteSet(1)


def test_original_domain_from_continuous_domain_can_feed_range(x):
    source = 1/(x*x - 1)
    domain = s.continuous_domain(source, x, s.Interval(-2, 2))
    expected = s.Union(s.Interval(-s.oo, -1), s.Interval(s.Rational(1, 3), s.oo))
    assert s.function_range(source, x, domain) == expected
    assert s.function_range(source, x, s.Interval(-2, 2)) == expected


def test_empty_intersection_has_no_image_or_extremum(x):
    domain = s.Intersection(s.FiniteSet(0), s.Interval.open(0, 1))
    assert s.function_range(x*x, x, domain) == s.S.EmptySet
    assert s.stationary_points(x*x, x, domain) == s.S.EmptySet
    for operation in (s.minimum, s.maximum):
        with pytest.raises(ValueError, match="empty domain"):
            operation(x*x, x, domain)


def test_infinite_finite_set_members_are_outside_the_real_partition(x):
    points = s.FiniteSet(-s.oo, 2, s.oo)
    domain = s.Intersection(s.S.Reals, points)
    assert s.function_range(x*x, x, domain) == s.FiniteSet(4)
    assert s.stationary_points(1, x, domain) == s.FiniteSet(2)
    punctured = s.Complement(s.Interval(0, 3), points)
    expected = s.Union(s.Interval.Ropen(0, 2), s.Interval.Lopen(2, 3))
    assert s.function_range(x, x, punctured) == expected


def test_nowhere_defined_original_source_has_no_finite_image(x):
    source = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    domain = s.Union(s.Interval(0, 1), s.FiniteSet(2))
    assert s.function_range(source, x, domain) == s.S.EmptySet
    assert s.stationary_points(source, x, domain) == s.S.EmptySet
    with pytest.raises(ValueError, match="empty domain"):
        s.minimum(source, x, domain)


def test_stationary_points_keep_all_compound_domain_restrictions(x):
    source = x**4/4 - x*x/2
    domain = s.Complement(s.S.Reals, s.FiniteSet(0))
    assert s.stationary_points(source, x, domain) == s.FiniteSet(-1, 1)
    assert s.stationary_points(source, x, s.FiniteSet(-1, 0, 2)) == s.FiniteSet(-1, 0)


def test_constant_stationary_set_is_the_entire_requested_domain(x):
    domain = s.Union(s.Interval.open(-2, -1), s.FiniteSet(0, 2))
    assert s.stationary_points(7, x, domain) == domain
    assert s.function_range(7, x, domain) == s.FiniteSet(7)


def test_decreasing_map_reverses_each_component_endpoint(x):
    domain = s.Union(s.Interval.Lopen(-3, -2), s.Interval.Ropen(0, 1))
    expected = s.Union(s.Interval.Lopen(-1, 0), s.Interval.Ropen(2, 3))
    assert s.function_range(-x, x, domain) == expected


def test_exact_nearby_finite_points_do_not_collapse(x):
    epsilon = s.Rational(1, 10**80)
    domain = s.FiniteSet(1, 1 + epsilon)
    expected = s.FiniteSet(1, 1 + 2*epsilon + epsilon*epsilon)
    assert s.function_range(x*x, x, domain) == expected


def test_indexed_real_algebraic_finite_points_evaluate_exactly(x):
    t = s.Symbol("compound_endpoint_t")
    positive = s.CRootOf(t*t - 2, t, 1)
    negative = s.CRootOf(t*t - 2, t, 0)
    assert s.function_range(x*x, x, s.FiniteSet(positive, negative)) == s.FiniteSet(2)


def test_equivalent_algebraic_endpoint_and_hole_share_one_point(x):
    t, u = s.symbols("compound_bound_t compound_bound_u")
    bound = s.CRootOf(t*t - 2, t, 1)
    hole = s.CRootOf(u*u - 2, u, 1)
    domain = s.Complement(s.Interval(bound, 2), s.FiniteSet(hole))
    assert s.function_range(x*x, x, domain) == s.Interval.Lopen(2, 4)
    assert s.minimum(x*x, x, domain) == 2


def test_compound_imageset_uses_checked_original_domains(x):
    source = s.Mul(x*x - 1, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
    domain = s.Union(s.Interval(0, 1), s.Interval(2, 3))
    expected = s.Union(s.Interval.Ropen(1, 2), s.Interval(3, 4))
    assert s.imageset(x, source, domain) == expected


@pytest.mark.parametrize("operation", [s.function_range, s.minimum, s.maximum])
def test_source_mutation_during_chart_check_prevents_publication(x, operation):
    source = s.Add(x*x, 1, evaluate=False)
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        source._args = (x, s.Integer(1))
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="source changed"):
            operation(source, x, domain)


def test_source_native_and_retained_arguments_must_agree_at_ingress(x):
    source = s.Add(x*x, 1, evaluate=False)
    source._args = (x, s.Integer(1))
    with pytest.raises(NotImplementedError, match="source.*(disagrees|does not match)"):
        s.function_range(source, x, s.Interval(0, 1))


@pytest.mark.parametrize("operation", [s.function_range, s.minimum, s.maximum, s.stationary_points])
def test_compound_domain_mutation_during_check_prevents_publication(x, operation):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    replacement = s.Interval(10, 11)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        domain._native_set = replacement._native_set
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="domain.*changed"):
            operation(x*x, x, domain)


def test_finite_python_elements_cannot_disagree_with_native_storage(x):
    domain = s.FiniteSet(1, 2)
    domain._surface_elements = (s.Integer(1), s.Integer(3))
    with pytest.raises(NotImplementedError, match="elements.*native storage"):
        s.function_range(x*x, x, domain)


def test_finite_surface_mutation_during_output_prevents_publication(x):
    domain = s.FiniteSet(1, 2)
    ranges = importlib.import_module("sympy.calculus._rational_range")
    original = ranges._as_set

    def mutate(*args):
        result = original(*args)
        domain._surface_elements = (s.Integer(1), s.Integer(3))
        return result

    with patch.object(ranges, "_as_set", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="domain.*changed"):
            s.function_range(x*x, x, domain)


def test_empty_image_rechecks_source_after_output_construction(x):
    source = s.Add(x*x, 1, evaluate=False)
    ranges = importlib.import_module("sympy.calculus._rational_range")
    original = ranges._as_set

    def mutate(*args):
        result = original(*args)
        source._args = (x, s.Integer(1))
        return result

    with patch.object(ranges, "_as_set", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="source changed"):
            s.function_range(source, x, s.S.EmptySet)


def test_modified_set_descriptor_is_never_invoked(x):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))

    def forbidden(_):
        raise AssertionError("modified set descriptor ran")

    with patch.object(s.Union, "args", property(forbidden)):
        with pytest.raises(NotImplementedError):
            s.function_range(x*x, x, domain)


@pytest.mark.parametrize("error", [s.EvaluationBudgetExceeded, s.EvaluationCancelled])
def test_native_operational_failure_is_terminal_for_compound_domains(x, error):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=error("forced terminal failure")) as checked, \
         patch.object(core, "diff_with_receipt",
                      side_effect=AssertionError("derivative ran after failure")):
        with pytest.raises(error, match="terminal failure"):
            s.function_range(x*x, x, domain)
    assert checked.call_count == 1


def test_compound_image_failure_does_not_become_a_formal_imageset(x):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=ValueError("forced verification failure")):
        with pytest.raises(NotImplementedError, match="verification failure"):
            s.imageset(x, x*x, domain)


def test_finite_domain_preflight_refuses_oversized_point_lists(x):
    domain = s.FiniteSet(*range(129))
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("oversized domain reached algebra")):
        with pytest.raises(NotImplementedError):
            s.function_range(x*x, x, domain)


def test_unknown_compound_bound_cannot_erase_a_stationary_source_hole(x):
    bound = s.Symbol("compound_unknown_bound")
    domain = s.Union(s.Interval(bound, 1), s.Interval(2, 3))
    source = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    core = importlib.import_module("sympy.core")
    with patch.object(core, "diff", side_effect=AssertionError("generic derivative fallback ran")):
        with pytest.raises(NotImplementedError, match="domain endpoints"):
            s.stationary_points(source, x, domain)


def test_unsupported_compound_image_preserves_the_formal_original_source(x):
    bound = s.Symbol("compound_formal_bound")
    domain = s.Union(s.Interval(bound, 1), s.Interval(2, 3))
    source = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    result = s.imageset(x, source, domain)
    assert type(result) is s.ImageSet
    assert result.lamda.expr.args == source.args


def test_elementary_range_composes_with_disconnected_rational_images(x):
    domain = s.Union(s.Interval(-2, -1), s.Interval(0, 1))
    expected = s.Union(s.Interval(s.exp(-2), s.exp(-1)), s.Interval(1, s.E))
    assert s.function_range(s.exp(x), x, domain) == expected
    assert s.minimum(s.exp(x), x, domain) == s.exp(-2)
    assert s.maximum(s.exp(x), x, domain) == s.E


def test_elementary_branch_restrictions_apply_to_every_component(x):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2), s.FiniteSet(4))
    expected = s.Union(s.Interval(0, s.log(2)), s.FiniteSet(s.log(4)))
    assert s.function_range(s.log(x), x, domain) == expected
    assert s.minimum(s.log(x), x, domain) == 0
    assert s.maximum(s.log(x), x, domain) == s.log(4)


def test_elementary_finite_point_image_keeps_the_real_branch(x):
    assert s.function_range(s.sqrt(x), x, s.FiniteSet(-1, 0, 4)) == s.FiniteSet(0, 2)


def test_elementary_fold_merges_both_disconnected_preimages(x):
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2))
    assert s.function_range(s.Abs(x), x, domain) == s.Interval(1, 2)
    assert s.function_range(s.cosh(x), x, domain) == s.Interval(s.cosh(1), s.cosh(2))


def test_identity_range_retains_the_original_indexed_algebraic_endpoint(x):
    t = s.Symbol("compound_identity_bound")
    root = s.CRootOf(t*t - 2, t, 1)
    domain = s.Interval(root, 2)
    assert s.function_range(x, x, domain) == domain
    assert s.minimum(x, x, domain) == root


def test_rational_affine_elementary_endpoints_keep_literal_exact_values(x):
    source = s.Rational(3, 2)*s.exp(x) + s.Rational(1, 3)
    domain = s.Interval(0, 1)
    expected = s.Interval(s.Rational(11, 6), s.Rational(3, 2)*s.E + s.Rational(1, 3))
    assert s.function_range(source, x, domain) == expected
    assert s.minimum(source, x, domain) == s.Rational(11, 6)
