"""Checked recursive QQ/ZZ division through the actual public extension.

Generator order, coefficient obstructions, promotion and public result shapes
were compared with isolated SymPy 1.14.0. Source mutation and resource cases
exercise the implementation's exact original-source admission contract.
"""

from math import comb

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import polytools
from sympy.polys.polyerrors import CoercionFailed, ExactQuotientFailed, PolynomialError, UnificationFailed


def expressions(pair):
    return tuple(value.as_expr() if isinstance(value, sp.Poly) else value for value in pair)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
@pytest.mark.parametrize("reverse", [False, True])
def test_whole_leading_coefficients_and_primary_generator_determine_the_result(domain, reverse):
    x, y = sp.symbols("recursive_order_x recursive_order_y")
    ring = (y, x) if reverse else (x, y)
    left = sp.Poly(x*y + 1, *ring, domain=domain)
    right = sp.Poly((y + 1)*x + 1, *ring, domain=domain)
    expected = (1, -x) if reverse else (0, x*y + 1)
    for pair in [left.div(right), sp.div(left, right)]:
        assert expressions(pair) == expected
        assert all(value.gens == ring and value.domain == domain for value in pair)
    assert sp.div(left.as_expr(), right.as_expr(), *ring, domain=domain) == expected


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_exact_nonlinear_coefficients_divide_in_three_variables(cls, domain):
    x, y, z = sp.symbols("recursive_exact_x recursive_exact_y recursive_exact_z")
    divisor = (y + z)*x + y*z + 1
    quotient = x*y - x*z + z
    left = cls(quotient*divisor + y + 1, x, y, z, domain=domain)
    right = sp.Poly(divisor, x, y, z, domain=domain)
    result = left.div(right)
    assert expressions(result) == (quotient, y + 1)
    assert all(type(value) is cls and value.gens == (x, y, z) and value.domain == domain
               for value in result)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_prior_exact_steps_survive_the_next_coefficient_obstruction(domain):
    x, y = sp.symbols("recursive_partial_x recursive_partial_y")
    left = 3*x**2*y + 2*x*y**2 + 4*y + 5
    right = 2*x*y + 3*y**2 + 1
    expected = ((0, left) if domain == "ZZ" else
                (3*x/2, -5*x*y**2/2 - 3*x/2 + 4*y + 5))
    assert sp.div(left, right, x, y, domain=domain) == expected


def test_strict_integer_division_stops_after_its_prior_integral_steps():
    x, y = sp.symbols("recursive_integer_x recursive_integer_y")
    left = sp.Poly(2*x**2 + 2*x + 1, x, y, domain="ZZ")
    right = sp.Poly(2*x + 1, x, y, domain="ZZ")
    strict = left.div(right, auto=False)
    promoted = left.div(right)
    assert expressions(strict) == (x, x + 1)
    assert all(value.domain == "ZZ" for value in strict)
    assert expressions(promoted) == (x + sp.Rational(1, 2), sp.Rational(1, 2))
    assert all(value.domain == "QQ" for value in promoted)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("fractional", [False, True])
def test_methods_promote_both_outputs_only_when_their_coefficients_require_it(domain, auto, fractional):
    x = sp.Symbol("recursive_promotion_x")
    left = sp.Poly(x**2 + 1, x, domain=domain)
    right = sp.Poly((2 if fractional else 1)*x + 1, x, domain=domain)
    if fractional and domain == "ZZ" and not auto:
        expected, output_domain = (0, x**2 + 1), "ZZ"
    elif fractional:
        expected, output_domain = (x/2 - sp.Rational(1, 4), sp.Rational(5, 4)), "QQ"
    else:
        expected, output_domain = (x - 1, 2), domain
    result = left.div(right, auto=auto)
    assert expressions(result) == expected
    assert all(value.domain == output_domain for value in result)


@pytest.mark.parametrize("options,strict", [
    ({}, False), ({"domain": "ZZ"}, True), ({"domain": "QQ"}, False),
    ({"auto": False}, True), ({"domain": "ZZ", "auto": True}, False),
    ({"domain": "QQ", "auto": False}, False),
])
def test_module_domain_and_auto_options_select_integer_or_rational_division(options, strict):
    x = sp.Symbol("recursive_options_x")
    left, right = x**2 + 1, 2*x + 1
    expected = (0, left) if strict else (x/2 - sp.Rational(1, 4), sp.Rational(5, 4))
    assert sp.div(left, right, x, **options) == expected
    result = sp.div(left, right, x, polys=True, **options)
    assert expressions(result) == expected
    assert all(type(value) is sp.Poly and value.domain == ("ZZ" if strict else "QQ") for value in result)


@pytest.mark.parametrize("degree", [65, 256])
def test_original_high_degree_sources_avoid_legacy_expansion_and_division_limits(degree, monkeypatch):
    ring = sp.symbols("recursive_high_0:8")
    x, y = ring[:2]
    axes = (x, y) if degree == 65 else ring
    left = sp.Poly(0, *axes, domain="ZZ")
    left._expr = sp.Pow(x + y if degree == 65 else x, degree, evaluate=False)
    right = sp.Poly(x + y if degree == 65 else x, *axes, domain="ZZ")
    original_source = _native_expr(left._expr)
    original = _native.poly_recursive_div_native
    calls = []

    def observe(native_left, *args, **kwargs):
        assert native_left is original_source
        calls.append(True)
        return original(native_left, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("recursive division used an unrelated legacy division pipeline")

    monkeypatch.setattr(_native, "poly_recursive_div_native", observe)
    monkeypatch.setattr(_native, "poly_multivariate_div_rem_native", forbidden)
    monkeypatch.setattr(_native, "poly_div_rem_native", forbidden)
    quotient, remainder = left.div(right)
    assert remainder.as_expr() == 0
    if degree == 65:
        expected = {(k, 64 - k): sp.Integer(comb(64, k)) for k in range(65)}
        assert dict(quotient.terms()) == expected
    else:
        assert quotient.as_expr() == x**255
    assert quotient.gens == remainder.gens == axes
    assert quotient.domain == remainder.domain == "ZZ"
    assert calls == [True]


@pytest.mark.parametrize("kind", ["zero", "constants", "lower_degree", "coefficient_divisor"])
def test_zero_constant_and_primary_constant_divisors(kind):
    x, y = sp.symbols("recursive_constant_x recursive_constant_y")
    if kind == "zero":
        left, right, expected, domain = 0, x + 1, (0, 0), "ZZ"
    elif kind == "constants":
        left, right, expected, domain = 6, 4, (sp.Rational(3, 2), 0), "QQ"
    elif kind == "lower_degree":
        left, right, expected, domain = x + 1, x**2 + 1, (0, x + 1), "ZZ"
    else:
        left, right, expected, domain = x*y + 1, y + 1, (0, x*y + 1), "ZZ"
    result = sp.Poly(left, x, y, domain="ZZ").div(sp.Poly(right, x, y, domain="ZZ"))
    assert expressions(result) == expected
    assert all(value.domain == domain for value in result)
    assert sp.div(6, 4, x, domain="ZZ") == (0, 6)


@pytest.mark.parametrize("method", [False, True])
def test_zero_divisors_raise_the_public_arithmetic_exception(method):
    x = sp.Symbol("recursive_zero_divisor_x")
    with pytest.raises(ZeroDivisionError):
        if method:
            sp.Poly(x + 1, x).div(sp.Poly(0, x))
        else:
            sp.div(x + 1, 0, x)


def test_two_polynomials_unify_in_the_first_ring_order():
    x, y = sp.symbols("recursive_union_x recursive_union_y")
    left = sp.PurePoly(x*y + 1, x, y)
    right = sp.Poly(x + y, y, x)
    for result in (left.div(right), sp.div(left, right)):
        assert expressions(result) == (y, 1 - y**2)
        assert all(type(value) is sp.PurePoly and value.gens == (x, y) for value in result)
    result = sp.Poly(x + 1, x).div(sp.Poly(y, y))
    assert expressions(result) == (0, x + 1)
    assert all(value.gens == (x, y) for value in result)


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("poly_on_left", [False, True])
def test_mixed_module_inputs_infer_ring_domain_and_base_poly_class(cls, poly_on_left):
    x = sp.Symbol("recursive_mixed_x")
    left = cls(2*x, x, domain="QQ") if poly_on_left else 2*x
    right = x if poly_on_left else cls(x, x, domain="QQ")
    result = sp.div(left, right)
    assert expressions(result) == (2, 0)
    assert all(type(value) is sp.Poly and value.gens == (x,) and value.domain == "ZZ" for value in result)
    assert sp.div(left, right, polys=False) == (2, 0)


def test_typed_same_name_generators_and_unused_dummy_axes_survive_lifting():
    ordinary = sp.Symbol("recursive_identity")
    positive = sp.Symbol("recursive_identity", positive=True)
    unused = sp.Dummy("recursive_identity")
    ring = (ordinary, positive, unused)
    divisor = ordinary - positive
    left = sp.Poly(ordinary*divisor + 1, *ring)
    right = sp.Poly(divisor, *ring)
    result = left.div(right)
    assert expressions(result) == (ordinary, 1)
    for value in result:
        assert all(actual is expected for actual, expected in zip(value.gens, ring))


@pytest.mark.parametrize("operation", ["div", "rem", "quo", "exquo"])
@pytest.mark.parametrize("method", [False, True])
def test_all_public_division_entries_share_the_exact_checked_result(operation, method):
    from sympy import polys

    x, y = sp.symbols("recursive_alias_x recursive_alias_y")
    divisor = x*y + 1
    left, right = sp.Poly((x + 1)*divisor, x, y), sp.Poly(divisor, x, y)
    assert getattr(sp, operation) is getattr(polys, operation) is getattr(polytools, operation)
    result = (getattr(left, operation)(right) if method else
              getattr(sp, operation)(left.as_expr(), right.as_expr(), x, y))
    pair = result if operation == "div" else (result,)
    expected = (x + 1, 0) if operation == "div" else (0,) if operation == "rem" else (x + 1,)
    assert expressions(pair) == expected
    if method:
        assert all(type(value) is sp.Poly and value.gens == (x, y) for value in pair)


@pytest.mark.parametrize("method", [False, True])
def test_exact_quotient_refuses_the_nonzero_recursive_remainder(method):
    x, y = sp.symbols("recursive_exquo_x recursive_exquo_y")
    left, right = x*y + 1, (y + 1)*x + 1
    with pytest.raises(ExactQuotientFailed):
        if method:
            sp.Poly(left, x, y).exquo(sp.Poly(right, x, y))
        else:
            sp.exquo(left, right, x, y)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_methods_accept_scalar_coefficients_but_reject_nonpoly_symbolic_operands(domain):
    x, y = sp.symbols("recursive_coercion_x recursive_coercion_y")
    source = sp.Poly(x + 1, x, domain=domain)
    assert expressions(source.div(2)) == (x/2 + sp.Rational(1, 2), 0)
    for operand in (x + 1, y):
        with pytest.raises(UnificationFailed):
            source.div(operand)
    if domain == "ZZ":
        with pytest.raises(UnificationFailed):
            source.div(sp.Rational(1, 2))
    else:
        assert expressions(source.div(sp.Rational(1, 2))) == (2*x + 2, 0)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_method_scalar_admission_does_not_erase_original_symbolic_children(domain):
    x, y = sp.symbols("recursive_scalar_source_x recursive_scalar_source_y")
    source = sp.Poly(x, x, y, domain=domain)
    held_scalar = sp.Add(x, -x, 1, evaluate=False)
    with pytest.raises(UnificationFailed):
        source.div(held_scalar)


@pytest.mark.parametrize("operation", ["div", "rem", "quo", "exquo"])
@pytest.mark.parametrize("domain,kind", [("ZZ", "fractional"), ("ZZ", "foreign"), ("QQ", "foreign")])
def test_original_coefficient_domains_cannot_be_repaired_by_generator_union(operation, domain, kind, monkeypatch):
    x, y = sp.symbols("recursive_invalid_x recursive_invalid_y")
    left = sp.Poly(x, x, domain=domain)
    left._expr = x/2 if kind == "fractional" else x*y
    right = sp.Poly(y, y, domain="QQ")

    def forbidden(*args, **kwargs):
        raise AssertionError("an invalid retained coefficient domain reached output lifting")

    monkeypatch.setattr(polytools, "_lift_native_result", forbidden)
    with pytest.raises((ValueError, CoercionFailed, PolynomialError)):
        getattr(sp, operation)(left, right, x, y, polys=False)


@pytest.mark.parametrize("kind", ["reciprocal", "function", "float"])
def test_original_unsupported_polynomial_children_survive_zero_shortcuts(kind):
    x, y = sp.symbols("recursive_hidden_x recursive_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False), "function": sp.sin(y),
           "float": sp.Float("1.0")}[kind]
    source = sp.Poly(0, x, y)
    source._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, TypeError, NotImplementedError, PolynomialError)):
        source.div(sp.Poly(1, x, y))


@pytest.mark.parametrize("hidden", [False, True])
def test_parametric_routing_cannot_repair_a_false_original_qq_polynomial(hidden):
    x, y = sp.symbols("recursive_original_ring_x recursive_original_ring_y")
    source = sp.Poly(0, x, y, domain="QQ")
    reciprocal = sp.Pow(y, -1, evaluate=False)
    source._expr = (sp.Mul(0, reciprocal, evaluate=False) if hidden
                    else sp.Mul(x, reciprocal, evaluate=False))
    divisor = sp.Poly(1, x, y, domain="QQ")
    with pytest.raises((ValueError, CoercionFailed, PolynomialError)):
        sp.div(source, divisor, x)


@pytest.mark.parametrize("target", ["left", "right", "unused_assumptions", "unused_identity", "generator_container"])
def test_original_operands_and_unused_axes_cannot_mutate_during_native_division(target, monkeypatch):
    x = sp.Symbol("recursive_mutation_x")
    y = sp.Symbol("recursive_mutation_y", positive=True)
    replacement = sp.Symbol("recursive_mutation_replacement")
    left, right = sp.Poly(x**2 - 1, x, y), sp.Poly(x + 1, x, y)
    generators = [x] if target in ("unused_assumptions", "unused_identity") else [x, y]
    original = _native.poly_recursive_div_native
    calls = []

    def mutate(*args, **kwargs):
        calls.append(True)
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

    monkeypatch.setattr(_native, "poly_recursive_div_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.div(left, right, generators)
    assert calls == [True]


@pytest.mark.parametrize("operation", ["div", "rem", "quo", "exquo"])
def test_native_refusal_is_terminal_without_a_legacy_retry(operation, monkeypatch):
    x, y = sp.symbols("recursive_terminal_x recursive_terminal_y")
    left, right = (x + y)*(x*y + 1), x + y
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError("injected recursive division verifier exhaustion")

    monkeypatch.setattr(_native, "poly_recursive_div_native", fail)
    with pytest.raises(ValueError, match="verifier exhaustion"):
        getattr(sp, operation)(left, right, x, y)
    assert calls == [True]


def test_direct_typed_bridge_checks_original_domains_before_publishing_values():
    x, y = sp.symbols("recursive_native_x recursive_native_y")
    variables = [_symbol_binding(x), _symbol_binding(y)]
    left, right = _native_expr(x*y + 1), _native_expr((y + 1)*x + 1)
    quotient, remainder, domain = _native.poly_recursive_div_native(left, right, variables, domain="QQ")
    assert polytools._lift_native_result(quotient, x, y) == 0
    assert polytools._lift_native_result(remainder, x, y) == x*y + 1
    assert domain == "QQ"
    for invalid in (variables[:1], variables + variables[:1]):
        with pytest.raises(ValueError):
            _native.poly_recursive_div_native(left, right, invalid)
    with pytest.raises(ValueError):
        _native.poly_recursive_div_native(left, right, variables,
                                         left_variables=variables[:1], left_domain="QQ")
    with pytest.raises(ValueError, match="noninteger"):
        _native.poly_recursive_div_native(_native_expr(x/2), right, variables,
                                         left_variables=variables, left_domain="ZZ")


def test_existing_explicit_main_variable_parameter_lane_remains_available():
    x, a = sp.symbols("recursive_param_x recursive_param_a")
    assert sp.div(x**2 + a, x + a, x) == (x - a, a**2 + a)
    left, right = sp.Poly(x**2 + a*x, x), sp.Poly(x + a, x)
    result = sp.div(left, right)
    assert expressions(result) == (x, 0)
    assert all(value.gens == (x,) for value in result)
