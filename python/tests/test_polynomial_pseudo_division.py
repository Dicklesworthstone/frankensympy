"""Checked pseudo-division through the actual native Python extension.

These fixtures compare exact coefficients, original ordered/typed rings,
classes and domains. They do not promote this bounded slice to a full profile.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import polytools
from sympy.polys.polyerrors import (
    CoercionFailed, ExactQuotientFailed, GeneratorsNeeded, PolynomialError, UnificationFailed,
)


def expressions(pair):
    return tuple(value.as_expr() if isinstance(value, sp.Poly) else value for value in pair)


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_pseudo_division_preserves_class_domain_and_nonunit_scaling(cls, domain):
    x = sp.Symbol("pseudo_basic_x")
    left, right = cls(x**2 + 1, x, domain=domain), sp.Poly(2*x - 4, x, domain=domain)
    for pair in (left.pdiv(right), sp.pdiv(left, right)):
        assert expressions(pair) == (2*x + 4, 20)
        assert all(type(value) is cls and value.gens == (x,) and value.domain == domain
                   for value in pair)
    assert sp.pdiv(x**2 + 1, 2*x - 4, x, domain=domain) == (2*x + 4, 20)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_symbolic_leading_coefficient_is_scaled_in_the_full_multivariate_ring(domain):
    x, y = sp.symbols("pseudo_coeff_x pseudo_coeff_y")
    left = sp.Poly(x**2 + 1, x, y, domain=domain)
    right = sp.Poly((y + 1)*x + y, x, y, domain=domain)
    result = left.pdiv(right)
    assert expressions(result) == (x*y + x - y, 2*y**2 + 2*y + 1)
    assert all(value.gens == (x, y) and value.domain == domain for value in result)
    assert sp.pdiv(left.as_expr(), right.as_expr(), x, y) == expressions(result)


@pytest.mark.parametrize("degree,divisor_degree", [(5, 4), (65, 64)])
def test_exact_early_termination_keeps_the_original_scaling_exponent(degree, divisor_degree, monkeypatch):
    x = sp.Symbol("pseudo_early_x")
    left = sp.Poly(0, x, domain="ZZ")
    left._expr = sp.Pow(x, degree, evaluate=False)
    right = sp.Poly(2*x**divisor_degree, x, domain="ZZ")
    native_source = _native_expr(left._expr)
    original = _native.poly_pseudo_div_native
    calls = []

    def observe(native_left, *args, **kwargs):
        assert native_left is native_source
        calls.append(True)
        return original(native_left, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("pseudo-division entered the ordinary division lane")

    monkeypatch.setattr(_native, "poly_pseudo_div_native", observe)
    monkeypatch.setattr(_native, "poly_recursive_div_native", forbidden)
    monkeypatch.setattr(_native, "poly_multivariate_div_rem_native", forbidden)
    monkeypatch.setattr(_native, "poly_div_rem_native", forbidden)
    assert expressions(left.pdiv(right)) == (2*x, 0)
    assert calls == [True]


def test_degree_skips_and_signed_leading_coefficients_have_exact_scaling():
    x = sp.Symbol("pseudo_skip_x")
    assert sp.pdiv(x**4 + 1, 3*x**2, x) == (9*x**2, 27)
    assert sp.pdiv(x**3 + 1, -2*x**2, x) == (-2*x, 4)
    assert sp.pdiv(x**256 + 1, x**256 + 2, x) == (1, -1)


@pytest.mark.parametrize("kind", ["zero", "constants", "lower", "coefficient"])
def test_zero_constant_and_primary_constant_cases(kind):
    x, y = sp.symbols("pseudo_constant_x pseudo_constant_y")
    if kind == "zero":
        left, right, expected = 0, 2*x + 1, (0, 0)
    elif kind == "constants":
        left, right, expected = 3, 2, (3, 0)
    elif kind == "lower":
        left, right, expected = x + y, x**2 + 1, (0, x + y)
    else:
        left, right, expected = x + 1, y + 1, (x*y + x + y + 1, 0)
    assert sp.pdiv(left, right, x, y) == expected
    result = sp.Poly(left, x, y, domain="ZZ").pdiv(sp.Poly(right, x, y, domain="ZZ"))
    assert expressions(result) == expected
    assert all(value.domain == "ZZ" for value in result)


def test_rational_coefficients_and_explicit_domain_are_retained():
    x = sp.Symbol("pseudo_rational_x")
    left, right = x**2/2 + sp.Rational(1, 3), x/2 + sp.Rational(1, 3)
    expected = (x/4 - sp.Rational(1, 6), sp.Rational(5, 36))
    assert sp.pdiv(left, right, x) == expected
    result = sp.pdiv(left, right, x, polys=True)
    assert expressions(result) == expected
    assert all(value.domain == "QQ" for value in result)
    with pytest.raises((ValueError, CoercionFailed, PolynomialError)):
        sp.pdiv(left, right, x, domain="ZZ")


@pytest.mark.parametrize("reverse", [False, True])
def test_named_polynomials_follow_the_first_operation_generator(reverse):
    x, y = sp.symbols("pseudo_order_x pseudo_order_y")
    axes = (y, x) if reverse else (x, y)
    left, right = sp.Poly(y**2 + x, *axes), sp.Poly(x*y + 1, *axes)
    expected = (x*y - 1, x**3 + 1) if reverse else (1, y**3 - 1)
    assert expressions(left.pdiv(right)) == expected
    assert sp.pdiv(left.as_expr(), right.as_expr(), axes) == expected
    assert sp.pdiv(left.as_expr(), right.as_expr(), gens=axes) == expected


def test_named_ring_unification_interleaves_unique_axes_before_shared_axes():
    x, y, z = sp.symbols("pseudo_union_x pseudo_union_y pseudo_union_z")
    left, right = sp.Poly(x**2 + 1, x), sp.Poly(x*y + 1, y, x)
    for result in (left.pdiv(right), sp.pdiv(left, right)):
        assert expressions(result) == (0, x**2 + 1)
        assert all(value.gens == (y, x) for value in result)
    left, right = sp.Poly(x + z, x, z), sp.Poly(y + z, y, z)
    result = left.pdiv(right)
    assert expressions(result) == (x*y + x*z + y*z + z**2, 0)
    assert all(value.gens == (x, y, z) for value in result)


def test_purepoly_same_name_distinct_assumptions_map_to_the_left_typed_owner():
    left_axis = sp.Symbol("pseudo_pure_identity", positive=True)
    right_axis = sp.Symbol("pseudo_pure_identity", real=True)
    left = sp.PurePoly(left_axis**2 + 1, left_axis)
    right = sp.Poly(2*right_axis - 4, right_axis)
    result = left.pdiv(right)
    assert expressions(result) == (2*left_axis + 4, 20)
    assert all(value.gens[0] is left_axis for value in result)
    assert result[0].as_expr().free_symbols == {left_axis}


@pytest.mark.parametrize("right_class", [sp.Poly, sp.PurePoly])
def test_purepoly_methods_unify_ordinal_coordinates_instead_of_symbol_names(right_class):
    x, z = sp.symbols("pseudo_pure_x pseudo_pure_z")
    left = sp.PurePoly(x**2 + 1, x, domain="QQ")
    right = right_class(2*z - 4, z, domain="ZZ")
    for result in (left.pdiv(right), sp.pdiv(left, right)):
        assert expressions(result) == (2*x + 4, 20)
        assert all(type(value) is sp.PurePoly and value.gens == (x,) and value.domain == "QQ"
                   for value in result)
    with pytest.raises(UnificationFailed, match="equal generator counts"):
        left.pdiv(right_class(z, z, x))


def test_purepoly_multivariate_coordinate_order_and_unused_axes_are_retained():
    x, y, z, w = sp.symbols("pseudo_pure_mv_x pseudo_pure_mv_y pseudo_pure_mv_z pseudo_pure_mv_w")
    left = sp.PurePoly(x**2 + y, x, y, domain="ZZ")
    right = sp.Poly(2*w - z, w, z, domain="ZZ")
    result = left.pdiv(right)
    assert expressions(result) == (2*x + y, y**2 + 4*y)
    assert all(value.gens == (x, y) and value.domain == "ZZ" for value in result)


@pytest.mark.parametrize("operation", ["pdiv", "prem", "pquo", "pexquo"])
@pytest.mark.parametrize("method", [False, True])
def test_all_public_entries_share_the_checked_scaled_result(operation, method):
    from sympy import polys

    x, y = sp.symbols("pseudo_alias_x pseudo_alias_y")
    divisor = x*y + 1
    left, right = sp.Poly((x + 1)*divisor, x, y), sp.Poly(divisor, x, y)
    assert getattr(sp, operation) is getattr(polys, operation) is getattr(polytools, operation)
    result = (getattr(left, operation)(right) if method else
              getattr(sp, operation)(left.as_expr(), right.as_expr(), x, y))
    pair = result if operation == "pdiv" else (result,)
    quotient = x*y**2 + y**2
    expected = (quotient, 0) if operation == "pdiv" else (0,) if operation == "prem" else (quotient,)
    assert expressions(pair) == expected
    if method:
        assert all(type(value) is sp.Poly and value.gens == (x, y) for value in pair)


@pytest.mark.parametrize("method", [False, True])
def test_exact_pseudo_quotient_refuses_a_nonzero_remainder(method):
    x = sp.Symbol("pseudo_exact_x")
    with pytest.raises(ExactQuotientFailed):
        if method:
            sp.Poly(x**2 + 1, x).pexquo(sp.Poly(2*x - 4, x))
        else:
            sp.pexquo(x**2 + 1, 2*x - 4, x)


@pytest.mark.parametrize("poly_on_left", [False, True])
def test_mixed_module_inputs_and_polys_controls_keep_the_expected_result_class(poly_on_left):
    x = sp.Symbol("pseudo_mixed_x")
    left = sp.PurePoly(x**2 + 1, x, domain="QQ") if poly_on_left else x**2 + 1
    right = 2*x - 4 if poly_on_left else sp.PurePoly(2*x - 4, x, domain="QQ")
    result = sp.pdiv(left, right)
    assert expressions(result) == (2*x + 4, 20)
    assert all(type(value) is sp.Poly and value.domain == "ZZ" for value in result)
    assert sp.pdiv(left, right, polys=False) == (2*x + 4, 20)
    result = sp.pdiv(x**2 + 1, 2*x - 4, x, domain="QQ", polys=True)
    assert all(type(value) is sp.Poly and value.domain == "QQ" for value in result)


def test_same_name_typed_symbols_and_unused_dummy_axes_survive_lifting():
    x = sp.Symbol("pseudo_identity")
    y = sp.Symbol("pseudo_identity", positive=True)
    unused = sp.Dummy("pseudo_identity")
    axes = (x, y, unused)
    result = sp.Poly(x**2 + y, *axes).pdiv(sp.Poly(y*x + 1, *axes))
    assert expressions(result) == (x*y - 1, y**3 + 1)
    assert all(all(actual is expected for actual, expected in zip(value.gens, axes))
               for value in result)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_scalar_methods_stay_in_the_original_coefficient_domain(domain):
    x, y = sp.symbols("pseudo_scalar_x pseudo_scalar_y")
    source = sp.Poly(x + 1, x, domain=domain)
    assert expressions(source.pdiv(2)) == (2*x + 2, 0)
    for operand in (x + 1, y, sp.Add(x, -x, 1, evaluate=False)):
        with pytest.raises(UnificationFailed):
            source.pdiv(operand)
    if domain == "ZZ":
        with pytest.raises(UnificationFailed):
            source.pdiv(sp.Rational(1, 2))
    else:
        assert expressions(source.pdiv(sp.Rational(1, 2))) == (x/2 + sp.Rational(1, 2), 0)


@pytest.mark.parametrize("operation", ["pdiv", "prem", "pquo", "pexquo"])
@pytest.mark.parametrize("kind", ["fractional", "foreign"])
def test_original_zz_declarations_cannot_be_repaired_by_requested_qq_or_ring_union(operation, kind):
    x, y = sp.symbols("pseudo_declaration_x pseudo_declaration_y")
    left = sp.Poly(0, x, domain="ZZ")
    left._expr = x/2 if kind == "fractional" else x*y
    right = sp.Poly(y, y, domain="QQ")
    with pytest.raises((ValueError, CoercionFailed, PolynomialError)):
        getattr(sp, operation)(left, right, x, y, domain="QQ", polys=False)


def test_purepoly_coordinate_renaming_cannot_repair_a_false_original_rhs_domain():
    x, y, z = sp.symbols("pseudo_bad_pure_x pseudo_bad_pure_y pseudo_bad_pure_z")
    left, right = sp.PurePoly(x**2, x), sp.Poly(0, z, domain="ZZ")
    right._expr = z*y
    with pytest.raises((ValueError, CoercionFailed, PolynomialError, NotImplementedError)):
        left.pdiv(right)


@pytest.mark.parametrize("kind", ["reciprocal", "function", "float"])
def test_original_unsupported_children_survive_zero_shortcuts(kind):
    x, y = sp.symbols("pseudo_hidden_x pseudo_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False), "function": sp.sin(y),
           "float": sp.Float("1.0")}[kind]
    source = sp.Poly(0, x, y)
    source._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, TypeError, NotImplementedError, PolynomialError)):
        source.pdiv(sp.Poly(1, x, y))


@pytest.mark.parametrize("target", ["left", "right", "unused_assumptions", "unused_identity", "generators"])
def test_sources_and_unused_axes_remain_immutable_through_native_publication(target, monkeypatch):
    x = sp.Symbol("pseudo_mutation_x")
    y = sp.Symbol("pseudo_mutation_y", positive=True)
    replacement = sp.Symbol("pseudo_mutation_replacement")
    left, right = sp.Poly(x**2 - 1, x, y), sp.Poly(2*x - 2, x, y)
    generators = [x] if target.startswith("unused") else [x, y]
    original = _native.poly_pseudo_div_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "left":
            monkeypatch.setattr(left, "_expr", x)
        elif target == "right":
            monkeypatch.setattr(right, "_gens", (y, x))
        elif target == "unused_assumptions":
            monkeypatch.setitem(y._assumptions, "positive", False)
        elif target == "unused_identity":
            monkeypatch.setattr(y, "_value", _native_expr(replacement))
        else:
            generators.reverse()
        return result

    monkeypatch.setattr(_native, "poly_pseudo_div_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.pdiv(left, right, generators)


def test_rhs_coordinate_owners_remain_retained_during_purepoly_mapping(monkeypatch):
    x, z, w = sp.symbols("pseudo_map_x pseudo_map_z pseudo_map_w")
    left, right = sp.PurePoly(x**2 + 1, x), sp.Poly(2*z - 4, z)
    original = _native.poly_pseudo_div_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(z, "_value", _native_expr(w))
        return result

    monkeypatch.setattr(_native, "poly_pseudo_div_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        left.pdiv(right)


def test_lifting_cannot_mutate_original_sources_after_native_acceptance(monkeypatch):
    x = sp.Symbol("pseudo_lift_x")
    left, right = sp.Poly(x**2 + 1, x), sp.Poly(2*x - 4, x)
    original = polytools._lift_native_result

    def mutate(value, *context):
        result = original(value, *context)
        monkeypatch.setattr(left, "_expr", x)
        return result

    monkeypatch.setattr(polytools, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        left.pdiv(right)


@pytest.mark.parametrize("operation", ["pdiv", "prem", "pquo", "pexquo"])
def test_native_refusal_is_terminal_for_every_entry(operation, monkeypatch):
    x = sp.Symbol("pseudo_terminal_x")
    calls = []

    def refuse(*args, **kwargs):
        calls.append(True)
        raise ValueError("injected pseudo-division verifier exhaustion")

    monkeypatch.setattr(_native, "poly_pseudo_div_native", refuse)
    with pytest.raises(ValueError, match="verifier exhaustion"):
        getattr(sp, operation)(x**2 + 1, 2*x - 4, x)
    assert calls == [True]


def test_direct_typed_bridge_admits_original_axes_before_positional_mapping():
    x, z = sp.symbols("pseudo_native_x pseudo_native_z")
    axes, right_axes = [_symbol_binding(x)], [_symbol_binding(z)]
    left, right = _native_expr(x**2 + 1), _native_expr(2*z - 4)
    quotient, remainder, domain = _native.poly_pseudo_div_native(
        left, right, axes, right_coordinates=right_axes,
        left_variables=axes, right_variables=right_axes, left_domain="QQ", right_domain="ZZ",
        domain="QQ",
    )
    assert polytools._lift_native_result(quotient, x, z) == 2*x + 4
    assert polytools._lift_native_result(remainder, x, z) == 20
    assert domain == "QQ"
    with pytest.raises(ValueError):
        _native.poly_pseudo_div_native(left, right, axes, right_coordinates=right_axes + axes)
    with pytest.raises(ValueError):
        _native.poly_pseudo_div_native(left, right, axes)
    with pytest.raises(ValueError):
        _native.poly_pseudo_div_native(
            _native_expr(x/2), _native_expr(x), axes, left_variables=axes, left_domain="ZZ",
        )


@pytest.mark.parametrize("method", [False, True])
def test_zero_divisor_is_not_a_partial_result(method):
    x = sp.Symbol("pseudo_zero_x")
    with pytest.raises(ZeroDivisionError):
        if method:
            sp.Poly(x, x).pdiv(sp.Poly(0, x))
        else:
            sp.pdiv(x, 0, x)


def test_unsupported_domains_parameters_options_and_growth_refuse_explicitly():
    x, y = sp.symbols("pseudo_refusal_x pseudo_refusal_y")
    with pytest.raises(GeneratorsNeeded):
        sp.pdiv(3, 2)
    with pytest.raises(NotImplementedError, match="complete polynomial ring"):
        sp.pdiv(x**2 + y, x*y + 1, x)
    with pytest.raises(NotImplementedError):
        sp.Poly(x, x, modulus=5).pdiv(sp.Poly(x, x, modulus=5))
    with pytest.raises(NotImplementedError):
        sp.pdiv(x, x, x, auto=True)
    with pytest.raises(TypeError):
        sp.pdiv(x, x, x, polys=1)
    with pytest.raises((ValueError, PolynomialError)):
        sp.pdiv(x**256, x*y + 1, x, y)
    with pytest.raises(UnificationFailed):
        sp.Poly(x*y, x).pdiv(sp.Poly(y, y))
