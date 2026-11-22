"""Public ordered QQ/ZZ reduction through the actual native extension.

Mathematical outputs, domain retraction, ordered divisors, and the two-Poly
fast path were fixed against isolated SymPy 1.14.0. Admission and mutation
cases enforce this implementation's original-source and resource contracts.
"""

from math import comb

import pytest
import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import _reduced
from sympy.polys.polyerrors import GeneratorsError, GeneratorsNeeded, PolynomialError, UnificationFailed


def test_standard_multivariate_reduction_is_available_without_a_groebner_search(monkeypatch):
    x, y = sp.symbols("reduced_basic_x reduced_basic_y")

    def forbidden(*args, **kwargs):
        raise AssertionError("ordered reduction must not construct or require a Groebner basis")

    monkeypatch.setattr(_native, "groebner_basis_native", forbidden)
    monkeypatch.setattr(_native, "groebner_reduce_native", forbidden)
    assert sp.reduced(x**2 + y**2, [x + y], x, y) == ([x - y], 2*y**2)
    assert sp.reduced(2*x**4 + y**2 - x**2 + y**3,
                      [x**3 - x, y**3 - y], x, y) == ([2*x, 1], x**2 + y**2 + y)


def test_arbitrary_divisor_order_changes_the_quotients_and_remainder():
    x, y = sp.symbols("reduced_order_x reduced_order_y")
    divisors = [x*y - 1, y**2 - 1]
    assert sp.reduced(x*y**2, divisors, x, y) == ([y, 0], y)
    assert sp.reduced(x*y**2, divisors[::-1], x, y) == ([x, 0], x)
    # A nonzero remainder for an arbitrary divisor list is not non-membership.
    assert sp.reduced(x - y, divisors, x, y) == ([0, 0], x - y)
    with pytest.raises(ValueError):
        _native.groebner_reduce_native(
            [_native_expr(value) for value in divisors],
            [_symbol_binding(x), _symbol_binding(y)], "lex", _native_expr(x - y),
        )


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_all_three_monomial_orders_match_their_distinct_reference_results(order):
    x, y, z = sp.symbols("reduced_orders_x reduced_orders_y reduced_orders_z")
    source = x*x*y + x*z*z + y**3 + z
    divisors = [x*y - z, y*y - x, z*z - y]
    expected = (([x, -z**2 - z, -y**2 - 2*y*z**2 - y*z - 2*z**4 - z**3],
                 2*z**6 + z**5 + z) if order == "lex" else
                ([x + 2, y, x], x*z + 3*z))
    assert sp.reduced(source, divisors, x, y, z, order=order) == expected


@pytest.mark.parametrize("options,strict", [
    ({}, False), ({"domain": "ZZ"}, True), ({"domain": "QQ"}, False),
    ({"auto": False}, True), ({"auto": True}, False),
    ({"domain": "ZZ", "auto": True}, False),
    ({"domain": "QQ", "auto": False}, False),
    ({"field": True}, False), ({"field": False}, True),
])
def test_integer_ring_and_automatic_field_extension_have_distinct_results(options, strict):
    x = sp.Symbol("reduced_integer_x")
    expected = ([0], x) if strict else ([sp.Rational(1, 2)], 0)
    assert sp.reduced(x, [2*x], x, **options) == expected
    quotients, remainder = sp.reduced(x, [2*x], x, polys=True, **options)
    assert ([value.as_expr() for value in quotients], remainder.as_expr()) == expected
    assert all(type(value) is sp.Poly and value.domain == ("ZZ" if strict else "QQ")
               for value in [*quotients, remainder])


def test_strict_integer_reduction_continues_past_each_coefficient_obstruction():
    x = sp.Symbol("reduced_obstruction_x")
    assert sp.reduced(3*x + 2, [2*x + 1, 3*x], x, domain="ZZ") == ([0, 1], 2)
    assert sp.reduced(x**2 + 2*x, [2*x], x, domain="ZZ") == ([1], x**2)


@pytest.mark.parametrize("options,domain", [
    ({}, "ZZ"), ({"auto": False}, "ZZ"), ({"domain": "ZZ"}, "ZZ"),
    ({"domain": "QQ"}, "QQ"), ({"field": True}, "QQ"),
])
def test_outputs_retract_together_only_when_the_original_operation_domain_allows_it(options, domain):
    x = sp.Symbol("reduced_retract_x")
    quotients, remainder = sp.reduced(x**2, [x], x, polys=True, **options)
    assert quotients[0].as_expr() == x and remainder.as_expr() == 0
    assert all(value.domain == domain for value in [*quotients, remainder])


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_two_polynomials_preserve_domains_but_public_results_use_poly(cls):
    x = sp.Symbol("reduced_polys_x")
    quotients, remainder = sp.reduced(cls(x*x, x, domain="QQ"), [sp.Poly(x, x, domain="ZZ")])
    assert quotients[0].as_expr() == x and remainder.as_expr() == 0
    assert all(type(value) is sp.Poly and value.domain == "QQ" and value.gens == (x,)
               for value in [*quotients, remainder])
    assert sp.reduced(cls(x*x, x), [sp.Poly(x, x)], polys=False) == ([x], 0)


@pytest.mark.parametrize("case", ["mixed_left", "mixed_right", "empty", "multiple"])
def test_general_expression_path_reinfers_domains_instead_of_using_two_poly_unification(case):
    x = sp.Symbol("reduced_inference_x")
    p = sp.Poly(x*x, x, domain="QQ")
    source, divisors = {
        "mixed_left": (p, [x]),
        "mixed_right": (x*x, [sp.Poly(x, x, domain="QQ")]),
        "empty": (p, []),
        "multiple": (p, [sp.Poly(x, x, domain="QQ"), sp.Poly(x + 1, x)]),
    }[case]
    quotients, remainder = sp.reduced(source, divisors)
    assert all(type(value) is sp.Poly and value.domain == "ZZ" and value.gens == (x,)
               for value in [*quotients, remainder])


def test_poly_ring_union_and_purepoly_positional_coordinates_are_not_conflated():
    x, y, z = sp.symbols("reduced_unify_x reduced_unify_y reduced_unify_z")
    quotients, remainder = sp.reduced(sp.Poly(x*y + 1, x, y), [sp.Poly(x + y, y, x)])
    assert quotients[0].as_expr() == y and remainder.as_expr() == 1 - y*y
    assert remainder.gens == (x, y)
    quotients, remainder = sp.reduced(sp.PurePoly(x*x + y, x, y), [sp.Poly(y + z, y, z)])
    assert quotients[0].as_expr() == x - y and remainder.as_expr() == y*y + y
    assert all(type(value) is sp.Poly and value.gens == (x, y) for value in [*quotients, remainder])
    quotients, remainder = sp.reduced(sp.PurePoly(x*x, x), [sp.PurePoly(y, y), sp.PurePoly(z, z)])
    assert [value.as_expr() for value in quotients] == [0, 0]
    assert remainder.as_expr() == x*x and remainder.gens == (x, y, z)
    with pytest.raises(UnificationFailed):
        sp.reduced(sp.PurePoly(x, x), [sp.PurePoly(y + z, y, z)])


def test_typed_same_name_symbols_and_unused_dummy_axes_survive_publication():
    x = sp.Symbol("reduced_same")
    positive = sp.Symbol("reduced_same", positive=True)
    unused = sp.Dummy("reduced_same")
    quotients, remainder = sp.reduced(positive*x + unused, [positive], positive, x, unused)
    assert quotients[0] is x and remainder is unused
    quotients, remainder = sp.reduced(
        sp.Poly(positive*x, positive, x, unused), [sp.Poly(positive, positive, x, unused)],
    )
    assert quotients[0].as_expr() is x
    assert all(all(actual is expected for actual, expected in zip(value.gens, (positive, x, unused)))
               for value in [*quotients, remainder])


def test_constant_zero_and_empty_divisor_results_and_zero_divisor_exception():
    x = sp.Symbol("reduced_constant_x")
    assert sp.reduced(0, [x], x) == ([0], 0)
    assert sp.reduced(x + 1, [], x) == ([], x + 1)
    assert sp.reduced(6, [4], x) == ([sp.Rational(3, 2)], 0)
    assert sp.reduced(6, [4], x, domain="ZZ") == ([0], 6)
    for source in [0, x]:
        with pytest.raises(ZeroDivisionError):
            sp.reduced(source, [x, 0], x)
    with pytest.raises(GeneratorsNeeded):
        sp.reduced(2, [3])


@pytest.mark.parametrize("keyword", [False, True])
def test_explicit_generator_containers_select_the_requested_ring_order(keyword):
    x, y = sp.symbols("reduced_axes_x reduced_axes_y")
    kwargs = {"gens": [y, x]} if keyword else {}
    args = () if keyword else ([y, x],)
    quotients, remainder = sp.reduced(y*y + x*x, [x + y], *args, polys=True, **kwargs)
    assert quotients[0].as_expr() == y - x and remainder.as_expr() == 2*x*x
    assert remainder.gens == (y, x)


def test_held_high_degree_source_reaches_native_admission_without_expansion():
    x, y = sp.symbols("reduced_held_x reduced_held_y")
    polynomial = sp.Poly(0, x, y, domain="ZZ")
    polynomial._expr = sp.Pow(x + y, 65, evaluate=False)
    source = _native_expr(polynomial._expr)
    quotients, remainder = sp.reduced(polynomial, [sp.Poly(x + y, x, y)])
    assert _native_expr(polynomial._expr) is source and remainder.as_expr() == 0
    expected = sp.Add(*(sp.Integer(comb(64, k))*x**k*y**(64 - k) for k in range(65)))
    # The native result is already a complete sparse expression. Compare all
    # 65 terms directly; requesting a second expansion obscures this contract.
    assert quotients[0].as_expr() == expected


@pytest.mark.parametrize("kind", ["pole", "zero_power", "hidden_function", "degree"])
def test_original_held_children_cannot_be_erased_to_admit_an_invalid_polynomial(kind):
    x = sp.Symbol("reduced_admission_" + kind)
    invalid = {
        "pole": lambda: sp.Mul(0, sp.Pow(x, -1, evaluate=False), evaluate=False),
        "zero_power": lambda: sp.Pow(sp.Pow(x, -1, evaluate=False), 0, evaluate=False),
        "hidden_function": lambda: sp.Mul(0, sp.sin(x), evaluate=False),
        "degree": lambda: sp.Mul(0, sp.Pow(x, 257, evaluate=False), evaluate=False),
    }[kind]()
    source = sp.Poly(0, x)
    source._expr = invalid
    with pytest.raises(ValueError):
        sp.reduced(source, [x], x)


def test_original_poly_ring_and_integer_domain_cannot_be_repaired_by_operation_unification():
    x, y = sp.symbols("reduced_original_x reduced_original_y")
    false_ring = sp.Poly(x + y, x, domain="QQ")
    with pytest.raises(ValueError):
        sp.reduced(false_ring, [x], x, y)
    with pytest.raises(ValueError):
        sp.reduced(sp.Poly(x/2, x, domain="ZZ"), [x], x, domain="QQ")
    noncommutative = sp.Symbol("reduced_unused_axis", commutative=False)
    with pytest.raises(PolynomialError):
        sp.reduced(sp.Poly(x, x, noncommutative), [x], x)


@pytest.mark.parametrize("target", ["divisors", "generators", "poly", "assumptions"])
def test_source_mutation_during_publication_refuses_the_whole_result(target, monkeypatch):
    x = sp.Symbol("reduced_mutation_" + target)
    polynomial = sp.Poly(x*x, x)
    divisors, axes = [x], [x]
    original = _reduced._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "divisors":
            divisors.append(x + 1)
        elif target == "generators":
            axes.append(sp.Symbol("reduced_added_axis"))
        elif target == "poly":
            polynomial._expr = x
        else:
            x._assumptions["positive"] = True
        return result

    monkeypatch.setattr(_reduced, "_lift_native_result", mutate)
    original_assumptions = dict(x._assumptions)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            sp.reduced(polynomial, divisors, axes)
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_assumptions)


def test_mutation_while_lifting_does_not_publish_a_partial_quotient_list(monkeypatch):
    x = sp.Symbol("reduced_lifting_x")
    polynomial = sp.Poly(x*x + 1, x)
    original = _reduced._lift_native_result

    def mutate(*args):
        result = original(*args)
        polynomial._gens = (sp.Symbol("reduced_changed_axis"),)
        return result

    monkeypatch.setattr(_reduced, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.reduced(polynomial, [x])


def test_unsupported_and_oversized_requests_refuse_before_native_reduction():
    x = sp.Symbol("reduced_limits_x")

    for values, axes, options, error in [
        ([x]*33, (x,), {}, ValueError),
        ([x], sp.symbols("reduced_toomany_0:9"), {}, GeneratorsError),
        ([x], (x, x), {}, GeneratorsError),
        ([x], (x,), {"domain": "RR"}, NotImplementedError),
        ([x], (x,), {"auto": 1}, TypeError),
        ([x], (x,), {"order": "invalid"}, ValueError),
    ]:
        with pytest.raises(error):
            sp.reduced(x, values, *axes, **options)
    with pytest.raises(TypeError):
        sp.reduced(x, iter([x]), x)
    with pytest.raises(ValueError):
        sp.reduced(1 << 4097, [x], x)


@pytest.mark.parametrize("target", ["hash", "equality", "constructor", "native"])
def test_replaced_public_implementations_are_refused_without_executing_them(target, monkeypatch):
    x = sp.Symbol("reduced_owner_" + target)
    calls = []

    def replaced(*args, **kwargs):
        calls.append(True)
        raise AssertionError("modified owner executed outside a supervised callback")

    if target == "hash":
        monkeypatch.setattr(sp.Symbol, "__hash__", replaced)
    elif target == "equality":
        monkeypatch.setattr(sp.Basic, "__eq__", replaced)
    elif target == "constructor":
        monkeypatch.setattr(sp.Poly, "__new__", replaced)
    else:
        monkeypatch.setattr(_native, "poly_reduced_native", replaced)
    with pytest.raises(TypeError, match="original"):
        sp.reduced(x, [x], x)
    assert calls == []


def test_native_bridge_checks_original_declaration_and_coordinate_dimensions():
    x, y = sp.symbols("reduced_native_x reduced_native_y")
    sources = [_native_expr(x), _native_expr(y)]
    axes = [_symbol_binding(x)]
    with pytest.raises(ValueError):
        _native.poly_reduced_native(sources, axes, [], [], [])
    with pytest.raises(ValueError):
        _native.poly_reduced_native(
            sources, axes, [axes, [_symbol_binding(y)]], ["QQ", "QQ"], [None, axes],
        )


def test_existing_prime_field_reduction_keeps_its_characteristic_and_lift_policy():
    x = sp.Symbol("reduced_prime_x")
    assert sp.reduced(x, [2*x], x, modulus=5) == ([-2], 0)
    quotients, remainder = sp.reduced(x, [2*x], x, modulus=5, symmetric=False, polys=True)
    assert quotients[0].as_expr() == 3 and remainder.as_expr() == 0
    assert all(value.get_modulus() == 5 for value in [*quotients, remainder])
