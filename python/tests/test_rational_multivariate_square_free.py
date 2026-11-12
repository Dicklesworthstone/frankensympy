"""Complete square-free groups through the actual QQ/ZZ native bridge.

Positive normalization and public result policies are pinned to isolated
SymPy 1.14.0. Admission and mutation cases exercise this implementation's
declared exact-source and bounded-resource contract.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import polytools
from sympy.polys.polyerrors import CoercionFailed, GeneratorsNeeded, PolynomialError


def factor_expressions(result):
    scale, factors = result
    return scale, [(factor.as_expr(), multiplicity) for factor, multiplicity in factors]


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_complete_nonlinear_groups_preserve_domain_and_class(cls, domain, monkeypatch):
    x, y = sp.symbols("sqf_group_x sqf_group_y")
    first, second = 2*x + 3*y, x*y + 1
    source = cls(12*first**2*second**3, x, y, domain=domain)

    def forbidden(*args, **kwargs):
        raise AssertionError("square-free decomposition must not require irreducible factorization")

    monkeypatch.setattr(polytools, "_native_factor_multivariate", forbidden)
    monkeypatch.setattr(_native, "poly_factor_multivariate_native", forbidden)
    monkeypatch.setattr(_native, "poly_factor_list_native", forbidden)
    expected_first = first if domain == "ZZ" else first/2
    expected_scale = 12 if domain == "ZZ" else 48
    scale, factors = source.sqf_list()
    assert scale == expected_scale
    assert [(factor.as_expr(), power) for factor, power in factors] == [
        (sp.expand(expected_first), 2), (second, 3)
    ]
    assert all(type(factor) is cls and factor.gens == (x, y) and factor.domain == domain
               for factor, _ in factors)
    part = source.sqf_part()
    assert type(part) is cls and part.gens == (x, y) and part.domain == domain
    assert part.as_expr() == sp.expand(expected_first*second)
    assert factor_expressions(sp.sqf_list(source)) == factor_expressions((scale, factors))
    assert sp.sqf_part(source) == part
    rebuilt = sp.sqf(source)
    assert type(rebuilt) is cls and rebuilt.gens == (x, y) and rebuilt.domain == "ZZ"
    assert rebuilt.as_expr() == sp.expand(12*first**2*second**3)


def test_all_partial_derivatives_detect_repeated_content_in_other_generators():
    x, y, z = sp.symbols("sqf_content_x sqf_content_y sqf_content_z")
    polynomial = sp.Poly(-18*y**3*(y + 1)**2, x, y, z, domain="ZZ")
    assert factor_expressions(polynomial.sqf_list()) == (-18, [(y + 1, 2), (y, 3)])
    part = polynomial.sqf_part()
    assert part.gens == (x, y, z)
    assert part.as_expr() == y**2 + y


def test_factors_with_the_same_multiplicity_remain_one_square_free_group():
    x, y = sp.symbols("sqf_equal_x sqf_equal_y")
    expression = (x + 1)**2*(y + 1)**2*(x + y)**3
    assert sp.sqf_list(expression, x, y) == (
        1, [(sp.expand((x + 1)*(y + 1)), 2), (x + y, 3)]
    )
    assert sp.sqf_part(expression, x, y) == sp.expand((x + 1)*(y + 1)*(x + y))


def test_high_degree_square_free_parts_do_not_need_irreducible_factorization(monkeypatch):
    x, y = sp.symbols("sqf_high_x sqf_high_y")
    factor = x**20 + y

    def forbidden(*args, **kwargs):
        raise AssertionError("complete factorization was consulted")

    monkeypatch.setattr(polytools, "_native_factor_multivariate", forbidden)
    assert sp.sqf_list(factor**2, x, y) == (1, [(factor, 2)])
    assert sp.sqf_part(factor**2, x, y) == factor


@pytest.mark.parametrize("domain", [None, "ZZ", "QQ"])
def test_rational_exterior_content_and_explicit_field_normalization(domain):
    x, y = sp.symbols("sqf_rational_x sqf_rational_y")
    factor = 2*x + 3*y
    expression = -sp.Rational(3, 5)*factor**2
    options = {} if domain is None else {"domain": domain}
    expected_factor = factor/2 if domain == "QQ" else factor
    expected_scale = -sp.Rational(12, 5) if domain == "QQ" else -sp.Rational(3, 5)
    assert sp.sqf_list(expression, x, y, **options) == (
        expected_scale, [(sp.expand(expected_factor), 2)]
    )
    as_poly = sp.sqf_list(expression, x, y, polys=True, **options)
    assert factor_expressions(as_poly) == (
        expected_scale, [(sp.expand(expected_factor), 2)]
    )
    assert as_poly[1][0][0].domain == ("QQ" if domain == "QQ" else "ZZ")
    assert sp.sqf(expression, x, y, **options) == expected_scale*sp.expand(expected_factor)**2
    if domain == "ZZ":
        with pytest.raises(CoercionFailed, match="noninteger"):
            sp.sqf_part(expression, x, y, **options)
    else:
        assert sp.sqf_part(expression, x, y, **options) == sp.expand(factor/2)


@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
@pytest.mark.parametrize("kind", ["zero", "constant", "polynomial"])
def test_square_free_part_discards_content_and_has_correct_zero_convention(domain, kind):
    x = sp.Symbol("sqf_univariate_content")
    expression = {"zero": 0, "constant": -7, "polynomial": -12*(2*x + 3)**3}[kind]
    expected = {"zero": 0, "constant": 1,
                "polynomial": 2*x + 3 if domain == "ZZ" else x + sp.Rational(3, 2)}[kind]
    polynomial = sp.Poly(expression, x, domain=domain)
    result = polynomial.sqf_part()
    assert result.as_expr() == expected
    assert result.gens == (x,) and result.domain == domain
    assert sp.sqf_part(polynomial, polys=False) == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_typed_identity_and_explicit_generator_order_survive_square_free_lifting(reverse):
    x = sp.Symbol("sqf_same_name", positive=True)
    y = sp.Symbol("sqf_same_name", negative=True)
    generators = (y, x) if reverse else (x, y)
    factor = 2*x + 3*y
    monic = factor/3 if reverse else factor/2
    source = sp.PurePoly(factor**2, *generators, domain="QQ")
    scale, factors = source.sqf_list()
    assert scale == (9 if reverse else 4)
    assert len(factors) == 1 and factors[0][1] == 2
    part = source.sqf_part()
    for result in (factors[0][0], part):
        assert type(result) is sp.PurePoly
        assert all(actual is expected for actual, expected in zip(result.gens, generators))
        assert result.as_expr() == sp.expand(monic)
        assert any(symbol is x for symbol in result.as_expr().free_symbols)
        assert any(symbol is y for symbol in result.as_expr().free_symbols)


def test_all_multiplicities_includes_only_the_missing_unit_groups():
    x, y = sp.symbols("sqf_all_x sqf_all_y")
    source = sp.Poly(6*x**2*y**4, x, y, domain="ZZ")
    assert factor_expressions(source.sqf_list(all=True)) == (
        6, [(1, 1), (x, 2), (1, 3), (y, 4)]
    )
    assert factor_expressions(source.sqf_list()) == (6, [(x, 2), (y, 4)])
    assert sp.Poly(0, x, y).sqf_list(all=True) == (0, [])
    assert sp.Poly(6, x, y).sqf_list(all=True) == (6, [])


@pytest.mark.parametrize("value", [0, -6, sp.Rational(3, 5)])
def test_scalar_expressions_and_explicit_constant_rings(value):
    x, y = sp.symbols("sqf_scalar_x sqf_scalar_y")
    assert sp.sqf_list(value) == (value, [])
    assert sp.sqf(value) == value
    with pytest.raises(GeneratorsNeeded):
        sp.sqf_part(value)
    assert sp.sqf_part(value, x, y) == (0 if value == 0 else 1)


@pytest.mark.parametrize("operation", ["sqf_list", "sqf_part", "sqf"])
@pytest.mark.parametrize("bad_kind", ["reciprocal", "function", "float"])
def test_original_unsupported_children_cannot_disappear_behind_zero(operation, bad_kind):
    x, y = sp.symbols("sqf_hidden_x sqf_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False),
           "function": sp.sin(y), "float": sp.Float("1.0")}[bad_kind]
    source = sp.Poly(0, x, y)
    source._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        getattr(sp, operation)(source)


@pytest.mark.parametrize("operation", ["sqf_list", "sqf_part", "sqf"])
@pytest.mark.parametrize("domain,kind,message", [
    ("ZZ", "fractional", "noninteger"),
    ("ZZ", "foreign_generator", "noninteger"),
    ("QQ", "foreign_generator", "nonrational"),
])
def test_original_coefficient_declarations_cannot_be_repaired_by_more_generators(
    operation, domain, kind, message, monkeypatch
):
    x, y = sp.symbols("sqf_invalid_x sqf_invalid_y")
    source = sp.Poly(x, x, domain=domain)
    source._expr = x/2 if kind == "fractional" else x*y

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid source reached result lifting")

    monkeypatch.setattr(polytools, "_lift_native_result", forbidden)
    with pytest.raises(CoercionFailed, match=message):
        getattr(sp, operation)(source, x, y)
    if operation != "sqf":
        with pytest.raises(CoercionFailed, match=message):
            getattr(sp, operation)(source, x, y, polys=False)


@pytest.mark.parametrize("target", ["metadata", "source", "generator_list", "assumptions"])
def test_source_and_generator_mutations_are_terminal(target, monkeypatch):
    x = sp.Symbol("sqf_mutation_x", positive=True)
    y = sp.Symbol("sqf_mutation_y")
    source = sp.Poly((x*y + 1)**2, x, y)
    generators = [x, y]
    original = _native.poly_multivariate_sqf_native
    calls = []

    def mutate(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        if target == "metadata":
            monkeypatch.setattr(source, "_gens", (y, x))
        elif target == "source":
            monkeypatch.setattr(source, "_expr", x)
        elif target == "generator_list":
            generators.reverse()
        else:
            monkeypatch.setitem(x._assumptions, "positive", False)
        return result

    monkeypatch.setattr(_native, "poly_multivariate_sqf_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.sqf_list(source, generators)
    assert calls == [True]


@pytest.mark.parametrize("operation", ["sqf_list", "sqf_part", "sqf"])
@pytest.mark.parametrize("target", ["assumptions", "identity"])
def test_unused_original_generators_remain_bound_when_the_operation_ring_changes(
    operation, target, monkeypatch
):
    x = sp.Symbol("sqf_retained_axis_x")
    y = sp.Symbol("sqf_retained_axis_y", positive=True)
    replacement = sp.Symbol("sqf_retained_axis_replacement")
    source = sp.Poly(x**2, x, y)
    original = _native.poly_multivariate_sqf_native
    calls = []

    def mutate(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        if target == "assumptions":
            monkeypatch.setitem(y._assumptions, "positive", False)
        else:
            monkeypatch.setattr(y, "_value", _native_expr(replacement))
        return result

    monkeypatch.setattr(_native, "poly_multivariate_sqf_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(sp, operation)(source, x)
    assert calls == [True]


@pytest.mark.parametrize("operation", ["sqf_list", "sqf_part", "sqf"])
def test_native_verification_or_budget_refusal_never_falls_back(operation, monkeypatch):
    x, y = sp.symbols("sqf_terminal_x sqf_terminal_y")
    source = (x + y)**2
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError("injected square-free verifier exhaustion")

    monkeypatch.setattr(_native, "poly_multivariate_sqf_native", fail)
    with pytest.raises(ValueError, match="verifier exhaustion"):
        getattr(sp, operation)(source, x, y)
    assert calls == [True]


@pytest.mark.parametrize("kind", ["custom_poly", "function", "noncommutative"])
def test_unsupported_source_owners_do_not_reach_native_decomposition(kind, monkeypatch):
    x = sp.Symbol("sqf_unsupported_x")

    class CustomPoly(sp.Poly):
        pass

    source = {"custom_poly": CustomPoly(x**2, x), "function": sp.sin(x),
              "noncommutative": sp.Symbol("sqf_noncommutative", commutative=False)**2}[kind]

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported source reached native decomposition")

    monkeypatch.setattr(_native, "poly_multivariate_sqf_native", forbidden)
    with pytest.raises((ValueError, TypeError, PolynomialError)):
        sp.sqf_list(source)


@pytest.mark.parametrize("normalization", ["ZZ", "QQ", "auto"])
def test_direct_typed_bridge_and_original_degree_bounds(normalization):
    x, y = sp.symbols("sqf_native_x sqf_native_y")
    source = 12*(2*x + y)**2
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    scale, factors, part, expanded, integral = _native.poly_multivariate_sqf_native(
        _native_expr(source), bindings, normalization
    )
    lift = lambda value: polytools._lift_native_result(value, source, x, y)
    expected_part = x + y/2 if normalization == "QQ" else 2*x + y
    assert integral is True
    assert lift(scale) == (48 if normalization == "QQ" else 12)
    assert [(lift(factor), power) for factor, power in factors] == [(expected_part, 2)]
    assert lift(part) == expected_part
    assert lift(expanded) == sp.expand(source)
    for invalid in [bindings + bindings[:1], bindings[:1]]:
        with pytest.raises(ValueError):
            _native.poly_multivariate_sqf_native(_native_expr(source), invalid, normalization)
    with pytest.raises(ValueError, match="normalization"):
        _native.poly_multivariate_sqf_native(_native_expr(source), bindings, "RR")
    held = sp.Pow(x + y, 100000, evaluate=False)
    with pytest.raises(ValueError, match="degree|bound"):
        _native.poly_multivariate_sqf_native(_native_expr(held), bindings, normalization)


def test_eight_axis_monomial_decomposition_retains_unused_generators():
    axes = sp.symbols("sqf_axis_0:8")
    source = sp.Poly(6*axes[0]**2*axes[-1]**3, *axes, domain="ZZ")
    scale, factors = source.sqf_list()
    assert scale == 6
    assert [(factor.as_expr(), power) for factor, power in factors] == [(axes[0], 2), (axes[-1], 3)]
    assert all(factor.gens == axes for factor, _ in factors)
    part = source.sqf_part()
    assert part.gens == axes and part.as_expr() == axes[0]*axes[-1]


@pytest.mark.parametrize("modulus", [5, 7])
@pytest.mark.parametrize("scale", [1, 2, -1, -2])
def test_prime_field_square_free_expression_entry_retains_its_polynomial_result(modulus, scale):
    x = sp.Symbol("sqf_modular_adjacent")
    source = sp.Poly(scale*(x + 1)**3, x, modulus=modulus)
    result = sp.sqf(source)
    assert type(result) is sp.Poly
    assert result.gens == (x,) and result.get_modulus() == modulus
    assert result == source
