"""Checked QQ/ZZ GCD and LCM through the actual public native bridge.

Expected contents and monic forms were compared with isolated SymPy 1.14.0.
The higher-degree fixture has coprime cofactors differing by one, so complete
factorization is unnecessary to know its exact greatest common divisor.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import polytools
from sympy.polys.polyerrors import PolynomialError


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_repeated_factors_are_complete_without_factorization(operation, monkeypatch):
    x, y = sp.symbols("qq_gcd_x qq_gcd_y")
    common = x*y + 1
    left = common**2*(x + 1)
    right = common*(y + 1)
    expected = common if operation == "gcd" else common**2*(x + 1)*(y + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("GCD/LCM must not depend on complete factorization")

    monkeypatch.setattr(polytools, "_native_factor_multivariate", forbidden)
    monkeypatch.setattr(_native, "poly_factor_multivariate_native", forbidden)
    monkeypatch.setattr(_native, "poly_factor_list_native", forbidden)
    for cls in (sp.Poly, sp.PurePoly):
        a, b = cls(left, x, y), cls(right, x, y)
        result = getattr(a, operation)(b)
        assert type(result) is cls
        assert result.as_expr() == sp.expand(expected)
        assert result.gens == (x, y)
        assert result.domain == "ZZ"
        assert getattr(b, operation)(a).as_expr() == sp.expand(expected)
    assert getattr(sp, operation)(left, right) == sp.expand(expected)


def test_high_degree_coprime_cofactors_need_no_irreducible_factorization(monkeypatch):
    x, y = sp.symbols("qq_high_x qq_high_y")
    common = x + y
    cofactor = x**10 + y**10 + 1

    def forbidden(*args, **kwargs):
        raise AssertionError("factorization was consulted")

    monkeypatch.setattr(polytools, "_native_factor_multivariate", forbidden)
    assert sp.gcd(common*cofactor, common*(cofactor + 1)) == common


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_integer_content_and_rational_monic_normalization(operation, domain):
    x, y = sp.symbols("qq_content_x qq_content_y")
    common = 2*x + 3*y
    left, right = 6*common*(x + 1), 9*common*(y + 1)
    scalar = (3 if operation == "gcd" else 18) if domain == "ZZ" else sp.Rational(1, 2)
    expected = scalar*common
    if operation == "lcm":
        expected *= (x + 1)*(y + 1)
    a, b = sp.Poly(left, x, y, domain=domain), sp.Poly(right, x, y, domain=domain)
    result = getattr(a, operation)(b)
    assert result.domain == domain
    assert result.as_expr() == sp.expand(expected)
    assert getattr(sp, operation)(left, right, x, y, domain=domain) == sp.expand(expected)
    as_poly = getattr(sp, operation)(left, right, x, y, domain=domain, polys=True)
    assert type(as_poly) is sp.Poly
    assert as_poly.domain == domain and as_poly.as_expr() == sp.expand(expected)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_rational_coefficients_and_mixed_domains_use_the_rational_field(operation):
    x, y = sp.symbols("qq_rational_x qq_rational_y")
    common = x*y + sp.Rational(2, 3)
    left = -sp.Rational(2, 3)*common*(x + sp.Rational(1, 2))
    right = sp.Rational(5, 7)*common*(y - sp.Rational(3, 7))
    expected = common if operation == "gcd" else common*(x + sp.Rational(1, 2))*(y - sp.Rational(3, 7))
    result = getattr(sp.Poly(left, x, y), operation)(sp.Poly(right, x, y))
    assert result.domain == "QQ"
    assert result.as_expr() == sp.expand(expected)
    assert getattr(sp, operation)(left, right) == sp.expand(expected)
    a = sp.Poly(6*x + 6*y, x, y, domain="ZZ")
    b = sp.Poly(9*x + 9*y, x, y, domain="QQ")
    mixed = getattr(a, operation)(b)
    assert mixed.domain == "QQ" and mixed.as_expr() == x + y
    with pytest.raises(ValueError, match="noninteger"):
        getattr(sp, operation)(left, right, x, y, domain="ZZ")


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_three_variable_ring_and_differing_input_generator_lists(operation):
    x, y, z = sp.symbols("qq_ring_x qq_ring_y qq_ring_z")
    common = x + y + z
    a, b = sp.Poly(common*(x + 1), x, y, z), sp.Poly(common*(y + 1), x, y, z)
    expected = common if operation == "gcd" else common*(x + 1)*(y + 1)
    assert getattr(a, operation)(b).as_expr() == sp.expand(expected)
    # Two separately univariate inputs still need the combined two-axis ring.
    result = getattr(sp.Poly(x, x), operation)(sp.Poly(y, y))
    assert result.gens == (x, y)
    assert result.as_expr() == (sp.Integer(1) if operation == "gcd" else x*y)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_full_typed_identity_and_generator_order_survive_lifting(operation):
    x = sp.Symbol("qq_same_name", positive=True)
    y = sp.Symbol("qq_same_name", negative=True)
    for gens, monic in [((x, y), x + 2*y), ((y, x), x/2 + y)]:
        common = x + 2*y
        left = sp.Poly(common*x, *gens, domain="QQ")
        right = sp.Poly(common*y, *gens, domain="QQ")
        expected = monic if operation == "gcd" else monic*x*y
        result = getattr(left, operation)(right)
        assert result.as_expr() == sp.expand(expected)
        assert all(actual is original for actual, original in zip(result.gens, gens))
        assert any(symbol is x for symbol in result.as_expr().free_symbols)
        assert any(symbol is y for symbol in result.as_expr().free_symbols)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_zero_and_scalar_inputs_preserve_the_declared_eight_axis_ring(operation, domain):
    axes = sp.symbols("qq_zero_0:8")
    source = sp.Poly(-2*axes[0] - 4*axes[-1], *axes, domain=domain)
    zero = sp.Poly(0, *axes, domain=domain)
    expected = 2*axes[0] + 4*axes[-1] if domain == "ZZ" else axes[0] + 2*axes[-1]
    result = getattr(source, operation)(zero)
    assert result.gens == axes and result.domain == domain
    assert result.as_expr() == (expected if operation == "gcd" else 0)
    assert getattr(zero, operation)(source).as_expr() == result.as_expr()
    assert getattr(zero, operation)(zero).as_expr() == 0
    constant = getattr(source, operation)(sp.Integer(2))
    assert constant.as_expr() == ((2 if domain == "ZZ" else 1)
                                  if operation == "gcd" else expected)


def test_scalar_rationals_keep_the_scalar_public_content_contract():
    assert sp.gcd(sp.Rational(2, 3), sp.Rational(4, 9)) == sp.Rational(2, 9)
    assert sp.lcm(sp.Rational(2, 3), sp.Rational(4, 9)) == sp.Rational(4, 3)
    assert sp.gcd(sp.Integer(0), sp.Integer(0)) == 0
    assert sp.lcm(sp.Integer(0), sp.Integer(0)) == 0


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("bad_kind", ["reciprocal", "function", "float"])
def test_original_unsupported_nodes_cannot_disappear_behind_zero(operation, bad_kind):
    x, y = sp.symbols("qq_hidden_x qq_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False),
           "function": sp.sin(y), "float": sp.Float("1.0")}[bad_kind]
    left = sp.Poly(0, x, y)
    right = sp.Poly(0, x, y)
    right._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        getattr(left, operation)(right)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_noncommutative_symbols_are_refused(operation):
    x = sp.Symbol("qq_noncommutative", commutative=False)
    y = sp.Symbol("qq_commutative")
    with pytest.raises(PolynomialError, match="noncommutative"):
        getattr(sp, operation)(x*y, x + y)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_source_metadata_mutation_and_native_failures_are_terminal(operation, monkeypatch):
    x, y = sp.symbols("qq_mutation_x qq_mutation_y")
    left, right = sp.Poly(x*x*y, x, y), sp.Poly(x*y*y, x, y)
    name = "poly_multivariate_" + operation + "_native"
    original = getattr(_native, name)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        right._gens = (y, x)
        return result

    monkeypatch.setattr(_native, name, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(left, operation)(right)
    right._gens = (x, y)

    def fail(*args, **kwargs):
        raise ValueError("test protected verifier budget exhausted")

    monkeypatch.setattr(_native, name, fail)
    with pytest.raises(ValueError, match="protected verifier budget exhausted"):
        getattr(left, operation)(right)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_both_native_entrypoints_and_ring_admission(operation):
    x, y = sp.symbols("qq_native_x qq_native_y")
    left, right = (x + y)*x, (x + y)*y
    expected = x + y if operation == "gcd" else sp.expand((x + y)*x*y)
    string_operation = getattr(_native, "poly_multivariate_" + operation + "_expr")
    typed_operation = getattr(_native, "poly_multivariate_" + operation + "_native")
    rendered = string_operation(str(left), str(right), [x.name, y.name])
    assert sp.expand(sp.sympify(rendered) - expected) == 0
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    raw = typed_operation(_native_expr(left), _native_expr(right), bindings)
    assert polytools._lift_native_result(raw, left, right, x, y) == expected
    for bad in [bindings + bindings[:1], [_symbol_binding(x)]]:
        with pytest.raises(ValueError):
            typed_operation(_native_expr(left), _native_expr(right), bad)
    with pytest.raises(ValueError, match="domain"):
        typed_operation(_native_expr(left), _native_expr(right), bindings, "RR")
    axes = sp.symbols("qq_auxiliary_0:8")
    with pytest.raises(ValueError, match="auxiliary"):
        sp.Poly(axes[0], *axes).gcd(sp.Poly(axes[-1], *axes))
