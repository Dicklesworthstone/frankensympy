"""Exact images and typed set identities (WS19 / WS21).

All range comparisons are exact. Held-domain cases exercise the project's
source-preserving contract; they do not claim full SymPy profile conformance.
"""

import importlib
from unittest.mock import patch

import pytest
import sympy as s


@pytest.mark.parametrize("scale", [10**40, 10**400])
def test_interval_image_does_not_collapse_distinct_exact_values(scale):
    x = s.Symbol("image_x")
    epsilon = s.Rational(1, scale)
    assert s.imageset(x, 1 + epsilon*x, s.Interval(0, 1)) == s.Interval(1, 1 + epsilon)


def test_huge_exact_image_never_converts_to_binary64():
    x = s.Symbol("image_large_x")
    offset = s.Integer(10**400)
    expression = offset + x
    domain = s.Interval(0, 1)
    with patch.object(s.Expr, "__float__", side_effect=AssertionError("float ordering ran")):
        result = s.imageset(s.Lambda(x, expression), domain)
    assert result == s.Interval(offset, offset + 1)


def test_complete_algebraic_critical_point_defines_exact_minimum():
    x = s.Symbol("image_quintic_x")
    source = x**6/6 - x*x/2 - x
    root = s.CRootOf(x**5 - x - 1, x, 0)
    from sympy.polys.rootoftools import _algebraic_comparison

    with patch.object(s, "solve", side_effect=AssertionError("incomplete solver ran")), \
         patch.object(s.Expr, "__float__", side_effect=AssertionError("float ordering ran")):
        result = s.imageset(x, source, s.Interval(1, 2))
    assert isinstance(result, s.Interval)
    assert _algebraic_comparison(result.start, source.subs(x, root)) == 0
    assert result.end == s.Rational(20, 3)
    assert not result.left_open and not result.right_open


@pytest.mark.parametrize("domain, expected", [
    (s.Interval.open(-1, 1), s.Interval.Ropen(0, 1)),
    (s.Interval.Lopen(0, 2), s.Interval.Lopen(0, 4)),
    (s.Interval.Ropen(0, 2), s.Interval.Ropen(0, 4)),
    (s.S.Reals, s.Interval(0, s.oo)),
])
def test_open_and_unbounded_images_keep_attained_extrema(domain, expected):
    x = s.Symbol("image_open_x")
    assert s.imageset(x, x*x, domain) == expected


def test_decreasing_image_reverses_endpoint_openness():
    x = s.Symbol("image_decreasing_x")
    assert s.imageset(x, -x, s.Interval.Lopen(0, 2)) == s.Interval.Ropen(-2, 0)


def test_constant_and_empty_images_are_exact_sets():
    x = s.Symbol("image_constant_x")
    assert s.imageset(x, 7, s.Interval.open(0, 1)) == s.FiniteSet(7)
    assert s.imageset(x, x*x, s.S.EmptySet) is s.S.EmptySet


def test_rational_poles_split_the_image():
    x = s.Symbol("image_pole_x")
    expected = s.Union(s.Interval(-s.oo, -1), s.Interval(1, s.oo))
    assert s.imageset(x, 1/x, s.Interval(-1, 1)) == expected
    assert s.imageset(x, 1/x**2, s.S.Reals) == s.Interval.open(0, s.oo)


def test_removable_held_hole_is_preserved():
    x = s.Symbol("image_held_x")
    source = s.Mul(x*x - 1, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
    expected = s.Union(s.Interval.Ropen(1, 2), s.Interval.Lopen(2, 3))
    assert s.imageset(x, source, s.Interval(0, 2)) == expected


def test_nested_reciprocal_keeps_its_original_exclusion():
    x = s.Symbol("image_nested_x")
    source = s.Pow(s.Pow(x, -1, evaluate=False), -1, evaluate=False)
    expected = s.Union(s.Interval.Ropen(-1, 0), s.Interval.Lopen(0, 1))
    assert s.imageset(x, source, s.Interval(-1, 1)) == expected


def test_algebraic_endpoint_uses_its_exact_selected_root():
    x, t = s.symbols("image_axis image_endpoint")
    root = s.CRootOf(t*t - 2, t, 1)
    result = s.imageset(x, x*x, s.Interval(0, root))
    assert result == s.Interval(0, 2)


def test_typed_coordinate_is_not_replaced_by_its_printed_name():
    plain = s.Symbol("image_same_name")
    declared = s.Symbol("image_same_name", positive=True)
    result = s.imageset(plain, declared**2, s.Interval(0, 2))
    assert isinstance(result, s.ImageSet)
    assert result.lamda.expr == declared**2
    dummy = s.Dummy("image_typed_dummy")
    assert s.imageset(dummy, dummy**2, s.Interval(-1, 1)) == s.Interval(0, 1)


def test_unsupported_grammar_remains_formal_before_native_work():
    x, parameter = s.symbols("image_formal_x image_parameter")
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=AssertionError("unsupported grammar reached a native check")):
        for expression in (s.sin(x), s.sqrt(x), parameter*x, s.Float("1.5")*x):
            assert isinstance(s.imageset(x, expression, s.Interval(1, 2)), s.ImageSet)
        assert isinstance(s.imageset(x, x*x, s.Interval(parameter, 2)), s.ImageSet)


def test_custom_function_is_not_differentiated_to_guess_a_range():
    x = s.Symbol("image_custom_x")

    class Custom(s.Function):
        def _eval_derivative(self, variable):
            raise AssertionError("custom derivative callback ran")

    expression = Custom(x, evaluate=False)
    assert isinstance(s.imageset(x, expression, s.Interval(0, 1)), s.ImageSet)


def test_native_failure_is_terminal_after_rational_selection():
    x = s.Symbol("image_failed_x")
    core = importlib.import_module("sympy.core")
    with patch.object(core._native, "solve_real_rational_inequality_native",
                      side_effect=ValueError("forced verification budget failure")) as checked, \
         patch.object(core, "diff_with_receipt",
                      side_effect=AssertionError("derivative ran after failed admission")):
        with pytest.raises(NotImplementedError, match="verification budget failure"):
            s.imageset(x, x*x, s.Interval(0, 1))
    assert checked.call_count == 1


def test_derivative_failure_is_not_replaced_with_a_formal_success():
    x = s.Symbol("image_derivative_x")
    core = importlib.import_module("sympy.core")
    with patch.object(core, "diff_with_receipt", side_effect=ValueError("derivation refused")):
        with pytest.raises(NotImplementedError, match="derivation refused"):
            s.imageset(x, x*x, s.Interval(0, 1))


def test_source_mutation_during_admission_prevents_image_publication():
    x = s.Symbol("image_mutation_x")
    source = s.Add(s.Pow(x, 2, evaluate=False), 1, evaluate=False)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        source._args = (x, s.Integer(1))
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="source changed"):
            s.imageset(x, source, s.Interval(0, 1))


def test_lambda_mutation_during_admission_prevents_image_publication():
    x = s.Symbol("image_lambda_mutation_x")
    function = s.Lambda(x, x*x)
    replacement = s.Lambda(x, x + 1)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        function._struct_args = replacement._struct_args
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="Lambda changed"):
            s.imageset(function, s.Interval(0, 1))


def test_changed_lambda_descriptor_refuses_without_invoking_it():
    x = s.Symbol("image_descriptor_x")
    function = s.Lambda(x, x*x)
    domain = s.Interval(0, 1)

    def forbidden(_):
        raise AssertionError("modified Lambda descriptor ran")

    with patch.object(s.Lambda, "expr", property(forbidden)):
        with pytest.raises(NotImplementedError, match="original Lambda"):
            s.imageset(function, domain)


def test_image_identity_preserves_free_symbol_assumptions():
    x = s.Symbol("image_identity_x")
    plain = s.Symbol("image_parameter")
    positive = s.Symbol("image_parameter", positive=True)
    first = s.ImageSet(s.Lambda(x, x + plain), s.S.Integers)
    second = s.ImageSet(s.Lambda(x, x + positive), s.S.Integers)
    assert str(first) == str(second)
    assert first != second
    assert len({first, second}) == 2
    assert len(s.Union(first, second).args) == 2


def test_image_identity_never_uses_printers():
    x = s.Symbol("image_unprinted_x")
    first = s.ImageSet(s.Lambda(x, x + 1), s.S.Integers)
    second = s.ImageSet(s.Lambda(x, x + 1), s.S.Integers)
    with patch.object(s.Lambda, "__str__", side_effect=AssertionError("printer used as identity")):
        assert first == second
        assert hash(first) == hash(second)


def test_finite_image_membership_retains_unknown_symbolic_equalities():
    x, a, b = s.symbols("image_finite_x image_finite_a image_finite_b")
    result = s.ImageSet(s.Lambda(x, x + a), s.FiniteSet(0, 1))
    assert result.contains(a) is True
    assert result.contains(b) is None
    with pytest.raises(TypeError):
        b in result


def test_multiple_finite_bases_use_simultaneous_lambda_arguments():
    x, y = s.symbols("image_multi_x image_multi_y")
    assert s.imageset(s.Lambda((x, y), x - y), s.FiniteSet(y), s.FiniteSet(x)) == s.FiniteSet(y - x)
    result = s.imageset(s.Lambda((x, y), x + y), s.Range(1, 3), s.FiniteSet(10, 20))
    assert result == s.FiniteSet(11, 12, 21, 22)


def test_image_construction_rejects_mismatched_arity():
    x, y = s.symbols("image_arity_x image_arity_y")
    with pytest.raises(ValueError, match="signature"):
        s.ImageSet(s.Lambda((x, y), x + y), s.S.Integers)


@pytest.mark.parametrize("value", [s.oo, -s.oo, s.zoo, s.nan])
def test_complex_numbers_do_not_include_nonfinite_atoms(value):
    assert s.S.Complexes.contains(value) is False


def test_complex_membership_preserves_known_and_unknown_facts():
    assert s.S.Complexes.contains(s.I) is True
    assert s.S.Complexes.contains(s.Rational(1, 3)) is True
    assert s.S.Complexes.contains(s.Symbol("image_unknown_complex")) is None


def test_callable_images_do_not_capture_a_same_named_free_symbol():
    free = s.Symbol("x")
    result = s.imageset(lambda argument: argument + free, s.FiniteSet(1, 2))
    assert result == s.FiniteSet(1 + free, 2 + free)


def test_lambda_signature_mutation_prevents_image_publication():
    x = s.Symbol("image_signature_x")
    function = s.Lambda(x, x*x)
    core = importlib.import_module("sympy.core")
    original = core._native.solve_real_rational_inequality_native

    def mutate(*args):
        result = original(*args)
        function.signature._args = (s.Symbol("image_replaced_axis"),)
        return result

    with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
        with pytest.raises(NotImplementedError, match="signature changed"):
            s.imageset(function, s.Interval(0, 1))


@pytest.mark.parametrize("value, expected", [
    (s.Float("0.5"), True),
    (s.Float("-1.25"), True),
    (s.Float("nan"), False),
    (s.Float("inf"), False),
    (s.Float("-inf"), False),
])
def test_complex_membership_distinguishes_finite_profile_floats(value, expected):
    assert s.S.Complexes.contains(value) is expected


def test_complex_membership_uses_certified_decimal_value_before_binary64_backing():
    from sympy.core import _ExactDecimalFloat

    value = _ExactDecimalFloat("1e1000", 15)
    assert s.S.Complexes.contains(value) is True
    assert s.S.Complexes.contains(s.Symbol("nan")) is None
