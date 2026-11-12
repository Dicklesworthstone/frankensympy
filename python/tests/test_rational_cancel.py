"""Original-source rational cancellation through the actual native extension.

Positive expression, generator-order and Poly identity cases were compared
with isolated SymPy 1.14.0. Held-source, mutation and terminal refusal cases
exercise FrankenSymPy's bounded exact-source contract.
"""

from math import comb

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import polytools
from sympy.polys.polyerrors import PolynomialError


def held_fraction(numerator, denominator):
    return sp.Mul(numerator, sp.Pow(denominator, -1, evaluate=False), evaluate=False)


def test_degree_65_original_source_reaches_one_checked_call_without_factorization(monkeypatch):
    x, y = sp.symbols("cancel_high_x cancel_high_y")
    source = held_fraction(sp.Pow(x + y, 65, evaluate=False), 2*x + 2*y)
    original_source = _native_expr(source)
    original_children = source.args
    original = _native.rational_cancel_native
    calls = []

    def observe(native_source, *args):
        assert native_source is original_source
        calls.append(True)
        return original(native_source, *args)

    def forbidden(*args, **kwargs):
        raise AssertionError("cancellation used a separate polynomial or factorization pipeline")

    monkeypatch.setattr(_native, "rational_cancel_native", observe)
    for name in ("primitive", "gcd", "div", "factor_list"):
        monkeypatch.setattr(sp.Poly, name, forbidden)
    monkeypatch.setattr(_native, "poly_factor_multivariate_native", forbidden)
    monkeypatch.setattr(_native, "poly_factor_list_native", forbidden)
    result = sp.cancel(source)
    expected = {(k, 64 - k): sp.Rational(comb(64, k), 2) for k in range(65)}
    assert dict(sp.Poly(result, x, y).terms()) == expected
    assert calls == [True]
    assert source.args is original_children


@pytest.mark.parametrize("case", ["repeated_nonlinear", "high_sparse_factor", "coprime_cofactors"])
def test_complete_multivariate_common_factors_do_not_require_irreducible_factors(case):
    x, y = sp.symbols("cancel_factor_x cancel_factor_y")
    common = x*y + 1
    if case == "repeated_nonlinear":
        source = held_fraction(common**2*(x + 1), common*(y + 1))
        expected = (x**2*y + x*y + x + 1)/(y + 1)
    elif case == "high_sparse_factor":
        common = x**20 + y
        source = held_fraction(common**2, common*(y + 1))
        expected = (x**20 + y)/(y + 1)
    else:
        source = held_fraction(common*(x**2 + y**2 + 1), common*(x + y + 1))
        expected = (x**2 + y**2 + 1)/(x + y + 1)
    assert sp.cancel(source) == expected


@pytest.mark.parametrize("case", ["rational_content", "joint_denominators", "constant_denominator", "monomial_denominator"])
def test_joint_primitive_normalization_preserves_the_exact_ratio(case):
    x, y = sp.symbols("cancel_content_x cancel_content_y")
    if case == "rational_content":
        source = held_fraction(sp.Rational(2, 3)*x + sp.Rational(4, 9)*y,
                               sp.Rational(5, 7)*x + sp.Rational(10, 21)*y)
        expected = sp.Rational(14, 15)
    elif case == "joint_denominators":
        source = held_fraction(sp.Rational(2, 3)*x + sp.Rational(4, 9)*y + sp.Rational(1, 2),
                               sp.Rational(5, 7)*x + sp.Rational(10, 21)*y + sp.Rational(2, 3))
        expected = (84*x + 56*y + 63)/(90*x + 60*y + 84)
    elif case == "constant_denominator":
        source = held_fraction(2*x + 3*y, 6)
        expected = x/3 + y/2
    else:
        source = held_fraction(2*x + 3*y, 6*x*y)
        expected = (2*x + 3*y)/(6*x*y)
    assert sp.cancel(source) == expected


@pytest.mark.parametrize("order", ["default", "x", "y", "z", "zy", "yx", "list", "tuple"])
def test_explicit_generator_order_and_remaining_parameters_determine_leading_sign(order):
    x, y, z = sp.symbols("x y z")
    gens = {"default": (), "x": (x,), "y": (y,), "z": (z,), "zy": (z, y),
            "yx": (y, x), "list": ([y, x],), "tuple": ((y, x),)}[order]
    source = held_fraction(x + 1, -2*x + 3*y)
    expected = ((-x - 1)/(2*x - 3*y) if order in ("default", "x", "z")
                else (x + 1)/(-2*x + 3*y))
    assert sp.cancel(source, *gens) == expected


@pytest.mark.parametrize("case", ["sum", "nested_inverse", "nested_zero", "polynomial"])
def test_common_denominators_nested_inverses_and_polynomial_expansion(case):
    x = sp.Symbol("cancel_composition_x")
    if case == "sum":
        source = sp.Add(1/(x + 1), 1/(x - 1), evaluate=False)
        expected = 2*x/(x**2 - 1)
    elif case == "nested_inverse":
        source = sp.Pow(held_fraction(x**2 - 1, x - 1), -1, evaluate=False)
        expected = 1/(x + 1)
    elif case == "nested_zero":
        source = sp.Mul(0, sp.Pow(x, -1, evaluate=False), evaluate=False)
        expected = 0
    else:
        source = sp.Mul(x, x + 1, evaluate=False)
        expected = x**2 + x
    assert sp.cancel(source) == expected


@pytest.mark.parametrize("value", [0, 1, -6, sp.Rational(3, 5)])
def test_constant_sources_use_the_same_checked_empty_ring(value):
    assert sp.cancel(value) == value


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", [None, "ZZ", "QQ"])
def test_polynomial_identity_preserves_original_object_ring_and_domain(cls, domain):
    x, y = sp.symbols("cancel_poly_x cancel_poly_y")
    kwargs = {} if domain is None else {"domain": domain}
    source = cls(6*x**2 + 3*x, y, x, **kwargs)
    original_domain = source.domain
    assert sp.cancel(source, object()) is source
    assert source.gens == (y, x) and source.domain == original_domain


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_prime_field_polynomial_identity_stays_in_its_original_field(cls, monkeypatch):
    x, y = sp.symbols("cancel_field_x cancel_field_y")
    source = cls(2*x*y + 3*x + 1, y, x, modulus=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("finite-field Poly identity entered rational cancellation")

    monkeypatch.setattr(_native, "rational_cancel_native", forbidden)
    assert sp.cancel(source, object()) is source
    assert source.gens == (y, x) and source.get_modulus() == 5


def test_same_printed_names_and_dummy_identity_do_not_merge():
    ordinary = sp.Symbol("cancel_identity")
    positive = sp.Symbol("cancel_identity", positive=True)
    dummy = sp.Dummy("cancel_identity")
    source = held_fraction((ordinary - positive)*(dummy + 1), ordinary - positive)
    result = sp.cancel(source)
    assert result == dummy + 1
    assert result.free_symbols == {dummy}
    assert next(iter(result.free_symbols)) is dummy


@pytest.mark.parametrize("domain,kind", [
    ("ZZ", "fractional"), ("ZZ", "foreign"), ("QQ", "foreign"), ("QQ", "reciprocal"),
])
def test_invalid_original_poly_domains_cannot_be_repaired_by_extra_generators(domain, kind, monkeypatch):
    x, y = sp.symbols("cancel_invalid_x cancel_invalid_y")
    source = sp.Poly(x, x, domain=domain)
    source._expr = x/2 if kind == "fractional" else x*y if kind == "foreign" else 1/x

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original Poly reached result lifting")

    monkeypatch.setattr(polytools, "_lift_native_result", forbidden)
    with pytest.raises(ValueError):
        sp.cancel(source, x, y)


@pytest.mark.parametrize("kind", ["zero_inverse", "zero_power", "cancelled_zero_denominator"])
def test_undefined_original_rational_children_cannot_disappear(kind):
    x = sp.Symbol("cancel_undefined_x")
    if kind == "zero_inverse":
        bad = sp.Pow(0, -1, evaluate=False)
        source = sp.Mul(0, bad, evaluate=False)
    elif kind == "zero_power":
        source = sp.Pow(sp.Pow(0, -1, evaluate=False), 0, evaluate=False)
    else:
        denominator = sp.Add(x, -x, evaluate=False)
        source = held_fraction(denominator, denominator)
    with pytest.raises(ValueError, match="zero|undefined"):
        sp.cancel(source)


@pytest.mark.parametrize("target", ["domain", "source", "unused_assumptions", "unused_identity"])
def test_original_poly_metadata_and_unused_axes_remain_bound(target, monkeypatch):
    x = sp.Symbol("cancel_mutation_x")
    y = sp.Symbol("cancel_mutation_y", positive=True)
    replacement = sp.Symbol("cancel_mutation_replacement")
    source = sp.Poly((x + 1)**2, x, y, domain="ZZ")
    original = _native.rational_cancel_native
    calls = []

    def mutate(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        if target == "domain":
            monkeypatch.setattr(source, "_domain", "QQ")
        elif target == "source":
            monkeypatch.setattr(source, "_expr", x)
        elif target == "unused_assumptions":
            monkeypatch.setitem(y._assumptions, "positive", False)
        else:
            monkeypatch.setattr(y, "_value", _native_expr(replacement))
        return result

    monkeypatch.setattr(_native, "rational_cancel_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.cancel(source)
    assert calls == [True]


@pytest.mark.parametrize("target", ["held_children", "generator_container", "lifting"])
def test_expression_and_request_mutations_are_terminal_through_publication(target, monkeypatch):
    x, y = sp.symbols("cancel_request_x cancel_request_y")
    source = held_fraction((x + y)**2, x + y)
    generators = [x, y]
    function = "_lift_native_result" if target == "lifting" else "rational_cancel_native"
    owner = polytools if target == "lifting" else _native
    original = getattr(owner, function)
    calls = []

    def mutate(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        if target == "generator_container":
            generators.reverse()
        else:
            monkeypatch.setattr(source, "_args", (x,))
        return result

    monkeypatch.setattr(owner, function, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.cancel(source, generators)
    assert calls == [True]


@pytest.mark.parametrize("operation", ["cancel", "simplify"])
@pytest.mark.parametrize("reason", ["verification refused", "generator budget exhausted", "cancelled"])
@pytest.mark.parametrize("form", ["rational", "polynomial"])
def test_native_failure_never_returns_a_partially_cancelled_result(operation, reason, form, monkeypatch):
    x, y = sp.symbols("cancel_terminal_x cancel_terminal_y")
    source = (held_fraction((x + y)**2, x + y) if form == "rational"
              else sp.Mul(x, x + 1, evaluate=False))
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError(reason)

    monkeypatch.setattr(_native, "rational_cancel_native", fail)
    with pytest.raises(ValueError, match=reason):
        getattr(sp, operation)(source)
    assert calls == [True]


def optional_cancellation_source(kind):
    x = sp.Symbol("cancel_optional_x")
    if kind == "high_nonrational_power":
        return sp.Pow(sp.sin(x), 257, evaluate=False)
    if kind == "nine_axes":
        ordinary = sp.Symbol("cancel_optional_axis")
        positive = sp.Symbol("cancel_optional_axis", positive=True)
        others = tuple(sp.Symbol(f"cancel_optional_extra_{index}") for index in range(7))
        source = sp.Add(ordinary, positive, *others)
        assert len(source.free_symbols) == 9
        return source
    if kind == "noncommutative":
        return sp.Symbol("cancel_optional_noncommutative", commutative=False) + 1
    if kind == "polynomial":
        return sp.Mul(x, x + 1, evaluate=False)
    if kind == "undefined_inverse":
        return sp.Pow(0, -1, evaluate=False)
    if kind == "power_bound":
        return sp.Pow(x, 257, evaluate=False)
    if kind == "nonrational":
        return sp.sin(x) + 1
    if kind == "composite_degree":
        y = sp.Symbol("cancel_optional_y")
        return x**256*y
    if kind == "expansion_bound":
        axes = tuple(sp.Symbol(f"cancel_optional_expansion_{index}") for index in range(8))
        return sp.Pow(sp.Add(*axes), 10, evaluate=False)
    raise AssertionError(f"unknown optional cancellation case: {kind}")


@pytest.mark.parametrize("kind", [
    "high_nonrational_power", "nine_axes", "composite_degree", "noncommutative",
])
def test_optional_simplify_cancellation_skips_ineligible_sources(kind, monkeypatch):
    from sympy.simplify.simplify import _shortest

    source = optional_cancellation_source(kind)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("an ineligible optional rewrite entered checked QQ cancellation")

    monkeypatch.setattr(_native, "rational_cancel_native", forbidden)
    result = _shortest(source, source) if kind == "noncommutative" else sp.simplify(source)
    assert result == source
    assert calls == []


@pytest.mark.parametrize("kind,native_expected,shell_expected", [
    ("polynomial", True, True),
    ("undefined_inverse", True, True),
    ("power_bound", False, False),
    ("nine_axes", False, False),
    ("nonrational", False, False),
    ("noncommutative", True, False),
    ("composite_degree", False, False),
    ("expansion_bound", False, False),
])
def test_optional_cancellation_selector_declines_shape_limits_without_admitting_sources(
    kind, native_expected, shell_expected,
):
    from sympy.simplify.simplify import _cancellation_candidate

    source = optional_cancellation_source(kind)
    assert _native.rational_cancel_candidate_native(_native_expr(source)) is native_expected
    assert _cancellation_candidate(source) is shell_expected


@pytest.mark.parametrize("kind", ["function", "float", "radical", "constant"])
@pytest.mark.parametrize("hold", ["product", "power"])
def test_nonrational_held_children_keep_the_exact_original_structure(kind, hold, monkeypatch):
    x = sp.Symbol("cancel_decline_x")
    child = {"function": sp.sin(x), "float": sp.Float("1.25"),
             "radical": sp.sqrt(x), "constant": sp.pi}[kind]
    source = (sp.Mul(0, child, evaluate=False) if hold == "product"
              else sp.Pow(child, 0, evaluate=False))
    children = source.args

    def forbidden(*args, **kwargs):
        raise AssertionError("a nonrational source entered checked QQ cancellation")

    monkeypatch.setattr(_native, "rational_cancel_native", forbidden)
    assert sp.cancel(source) is source
    assert source.args is children and any(value is child for value in children)


def test_closed_nonrational_fraction_still_passes_through_general_simplify(monkeypatch):
    x = sp.Symbol("cancel_closed_x")
    source = sp.sin(x)/(x + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("a nonrational source entered checked QQ cancellation")

    monkeypatch.setattr(_native, "rational_cancel_native", forbidden)
    assert sp.cancel(source) is source
    assert sp.simplify(source) is source


@pytest.mark.parametrize("kind", ["power", "integer", "depth"])
def test_nonrational_routing_still_admits_all_original_source_bounds(kind):
    x = sp.Symbol("cancel_bound_x")
    if kind == "power":
        large = sp.Pow(x, 100000, evaluate=False)
    elif kind == "integer":
        large = sp.Integer(1 << 4097)
    else:
        large = x
        for _ in range(70):
            large = sp.Pow(large, 2, evaluate=False)
    source = sp.Mul(0, sp.sin(x), large, evaluate=False)
    with pytest.raises(ValueError, match="bound|bits|envelope"):
        sp.cancel(source)


def test_nonrational_identity_return_rejects_forged_original_children():
    x = sp.Symbol("cancel_forged_x")
    source = sp.Mul(0, sp.sin(x), evaluate=False)
    source._args = (sp.Integer(0), sp.cos(x))
    with pytest.raises(ValueError, match="retained surface arguments"):
        sp.cancel(source)


def test_nonrational_identity_return_checks_mutation_during_routing(monkeypatch):
    x = sp.Symbol("cancel_route_mutation_x")
    source = sp.Mul(0, sp.sin(x), evaluate=False)
    original = _native.rational_cancel_source_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(source, "_args", (sp.Integer(0), x))
        return result

    monkeypatch.setattr(_native, "rational_cancel_source_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.cancel(source)


@pytest.mark.parametrize("target", ["assumptions", "identity"])
def test_native_owned_nonrational_sources_bind_original_symbol_owners(target, monkeypatch):
    x = sp.Symbol("cancel_native_owner_x", positive=True)
    replacement = sp.Symbol("cancel_native_owner_replacement")
    source = sp.sin(x)/(x + 1)
    original = _native.rational_cancel_source_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "assumptions":
            monkeypatch.setitem(x._assumptions, "positive", False)
        else:
            monkeypatch.setattr(x, "_value", _native_expr(replacement))
        return result

    monkeypatch.setattr(_native, "rational_cancel_source_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.cancel(source)


def test_every_borrowed_source_is_bounded_before_capability_scans(monkeypatch):
    x = sp.Symbol("cancel_route_order_x")
    source = sp.sin(x)/(x + 1)
    native_source = _native_expr(source)
    original_preflight = _native.rational_cancel_source_native
    original_lower = polytools._native_expr
    admitted = []

    def preflight(value):
        result = original_preflight(value)
        admitted.append(value)
        return result

    def lower(value):
        if value is source:
            assert any(native_source is previous for previous in admitted)
        return original_lower(value)

    monkeypatch.setattr(_native, "rational_cancel_source_native", preflight)
    monkeypatch.setattr(polytools, "_native_expr", lower)
    assert sp.cancel(source, x) is source


def test_noncommutative_and_custom_poly_sources_do_not_enter_the_qq_lane(monkeypatch):
    x = sp.Symbol("cancel_noncommutative", commutative=False)
    ordinary = sp.Symbol("cancel_custom_poly")
    callbacks = []

    class CustomPoly(sp.Poly):
        def as_expr(self):
            callbacks.append(True)
            raise AssertionError("custom Poly protocol was invoked")

    custom = CustomPoly(ordinary**2, ordinary)

    def forbidden(*args, **kwargs):
        raise AssertionError("an unsupported source entered checked QQ cancellation")

    monkeypatch.setattr(_native, "rational_cancel_native", forbidden)
    with pytest.raises(PolynomialError, match="noncommutative"):
        sp.cancel(x)
    with pytest.raises(TypeError, match="exact Poly"):
        sp.cancel(custom)
    assert callbacks == []


def test_original_opaque_function_lane_keeps_its_owned_applications():
    class Law(sp.Function):
        pass

    x = sp.Symbol("cancel_opaque_x")
    law = Law(x)
    result = sp.cancel((law**2 - 1)/(law + 1))
    assert result == law - 1
    assert any(child is law for child in result.args)


def test_direct_typed_bridge_preserves_domain_and_declines_no_operational_failure():
    x, y = sp.symbols("cancel_native_x cancel_native_y")
    source = held_fraction((x*y + 1)**2, 2*(x*y + 1)*(y + 1))
    variables = [_symbol_binding(x), _symbol_binding(y)]
    result = _native.rational_cancel_native(_native_expr(source), variables)
    assert polytools._lift_native_result(result, source, x, y) == (x*y + 1)/(2*y + 2)
    for invalid in [variables[:1], variables + variables[:1]]:
        with pytest.raises(ValueError):
            _native.rational_cancel_native(_native_expr(source), invalid)
    with pytest.raises(ValueError, match="domain"):
        _native.rational_cancel_native(_native_expr(source), variables, "GF(5)")
    with pytest.raises(ValueError, match="noninteger"):
        _native.rational_cancel_native(_native_expr(x/2), variables, "ZZ")
    with pytest.raises(ValueError):
        _native.rational_cancel_native(_native_expr(sp.sin(x)/(x + 1)), variables)


def test_simplify_and_ratsimp_use_complete_checked_cancellation():
    x, y = sp.symbols("cancel_simplify_x cancel_simplify_y")
    common = x*y + 1
    source = held_fraction(common**2*(x + 1), common*(y + 1))
    expected = (x**2*y + x*y + x + 1)/(y + 1)
    assert sp.simplify(source) == expected
    assert sp.ratsimp(1/(x + 1) + 1/(x - 1)) == 2*x/(x**2 - 1)
