"""Complete prime-field GCD/LCM through the actual checked native extension.

Expected factors follow exact multiplicities, with separate SymPy 1.14.0
oracle fixtures. Source mutation and domain failures cannot trigger fallback.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import _finitefield
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_repeated_factors_have_complete_multiplicities(prime, operation):
    x, y = sp.symbols("field_gcd_x field_gcd_y")
    h = x + y
    left = sp.Poly(h**2*(x + 1), x, y, modulus=prime)
    right = sp.Poly(h*(y + 1), x, y, modulus=prime)
    expected = sp.Poly(h if operation == "gcd" else h**2*(x + 1)*(y + 1),
                       x, y, modulus=prime)
    result = getattr(left, operation)(right)
    assert result == expected
    assert getattr(right, operation)(left) == expected
    assert type(result) is sp.Poly and result.gens == (x, y)
    assert result.gens[0] is x and result.gens[1] is y
    assert result.get_modulus() == prime


@pytest.mark.parametrize("prime", [2, 3, 7])
def test_characteristic_and_field_polynomial_identity_are_preserved(prime):
    x, y = sp.symbols("field_characteristic_x field_characteristic_y")
    left = sp.Poly(x*x + y*y, x, y, modulus=prime)
    right = sp.Poly(x + y, x, y, modulus=prime)
    assert left.gcd(right) == (right if prime == 2 else sp.Poly(1, x, y, modulus=prime))
    assert left.lcm(right) == (left if prime == 2 else left*right)
    field_zero = sp.Poly(x**prime - x, x, y, modulus=prime)
    y_poly = sp.Poly(y, x, y, modulus=prime)
    assert field_zero.gcd(y_poly) == sp.Poly(1, x, y, modulus=prime)
    assert field_zero.lcm(y_poly) == field_zero*y_poly


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_poly_purepoly_module_entries_keep_typed_axes_and_lift_policy(operation, symmetric):
    x = sp.Symbol("field_gcd_binding", positive=True)
    y = sp.Dummy("field_gcd_binding")
    h = x + 4*y
    for cls in (sp.Poly, sp.PurePoly):
        left = cls(2*h*x, x, y, modulus=5, symmetric=symmetric)
        right = cls(3*h*y, x, y, modulus=5, symmetric=symmetric)
        expected = cls(h if operation == "gcd" else h*x*y,
                       x, y, modulus=5, symmetric=symmetric)
        result = getattr(left, operation)(right)
        assert type(result) is cls
        assert result == expected and result._symmetric is symmetric
        assert result.gens[0] is x and result.gens[1] is y
        assert getattr(sp, operation)(left, right) == expected
        assert getattr(sp, operation)(2*h*x, 3*h*y, x, y, modulus=5,
                                      symmetric=symmetric) == expected.as_expr()
        polynomial = getattr(sp, operation)(2*h*x, 3*h*y, x, y, modulus=5,
                                            symmetric=symmetric, polys=True)
        assert type(polynomial) is sp.Poly and polynomial == expected


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_generator_order_determines_monic_normalization(operation):
    x, y = sp.symbols("field_order_x field_order_y")
    for gens, normalized in [((x, y), x + 2*y), ((y, x), 3*x + y)]:
        left = sp.Poly((x + 2*y)*x, *gens, modulus=5)
        right = sp.Poly((x + 2*y)*y, *gens, modulus=5)
        expected = normalized if operation == "gcd" else normalized*x*y
        assert getattr(left, operation)(right) == sp.Poly(expected, *gens, modulus=5)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_zero_and_scalar_inputs_preserve_full_declared_ring(operation):
    variables = sp.symbols("field_gcd_zero_0:8")
    source = sp.Poly(2*variables[0] + 4*variables[-1], *variables, modulus=5)
    zero = sp.Poly(0, *variables, modulus=5)
    one = sp.Poly(1, *variables, modulus=5)
    normalized = sp.Poly(variables[0] + 2*variables[-1], *variables, modulus=5)
    assert getattr(zero, operation)(zero) == zero
    assert getattr(source, operation)(zero) == (normalized if operation == "gcd" else zero)
    assert getattr(zero, operation)(source) == (normalized if operation == "gcd" else zero)
    assert getattr(source, operation)(2) == (one if operation == "gcd" else normalized)
    assert getattr(one, operation)(source) == (one if operation == "gcd" else normalized)
    x, y = sp.symbols("field_gcd_nonconstant_x field_gcd_nonconstant_y")
    # The public constant field returned by complete elimination remains a
    # valid ring; it must not infer or manufacture a variable during GCD.
    constant = sp.groebner([x, y], x, y, modulus=5, polys=True).eliminate(x, y)
    empty = constant.reduce(1)[1]
    assert empty.gens == ()
    assert getattr(empty, operation)(2).gens == ()


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_new_operations_never_call_rational_or_univariate_fallbacks(monkeypatch, operation):
    x, y = sp.symbols("field_gcd_route_x field_gcd_route_y")
    left = sp.Poly(x*x*y, x, y, modulus=5)
    right = sp.Poly(x*y*y, x, y, modulus=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("multivariate GF(p) reached a rational or univariate fallback")

    for name in ("prime_field_polynomial_native", "poly_multivariate_gcd_native",
                 "poly_multivariate_lcm_native", "poly_factor_multivariate_native"):
        monkeypatch.setattr(_native, name, forbidden)
    result = getattr(left, operation)(right)
    expected = x*y if operation == "gcd" else x*x*y*y
    assert result == sp.Poly(expected, x, y, modulus=5)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_mixed_domains_and_typed_orders_are_terminal(operation):
    x, y = sp.symbols("field_gcd_mixed_x field_gcd_mixed_y")
    shadow = sp.Symbol("field_gcd_mixed_x", positive=True)
    left = sp.Poly(x*y, x, y, modulus=5)
    for other in [sp.Poly(x*y, x, y), sp.Poly(x*y, x, y, modulus=7),
                  sp.Poly(x*y, y, x, modulus=5), sp.Poly(shadow*y, shadow, y, modulus=5)]:
        with pytest.raises(UnificationFailed):
            getattr(left, operation)(other)
    with pytest.raises(UnificationFailed):
        getattr(left, operation)(x + y)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("kind", ["reciprocal", "float", "function"])
def test_original_undefined_or_unsupported_nodes_cannot_hide_behind_zero(operation, kind):
    x, y = sp.symbols("field_gcd_hidden_x field_gcd_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False),
           "float": sp.Float("1.0"), "function": sp.sin(y)}[kind]
    left = sp.Poly(0, x, y, modulus=5)
    right = sp.Poly(0, x, y, modulus=5)
    right._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, CoercionFailed)):
        getattr(left, operation)(right)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("field,replacement", [("_modulus", 7), ("_symmetric", False)])
def test_both_original_metadata_snapshots_survive_native_verification(
        monkeypatch, operation, side, field, replacement):
    x, y = sp.symbols("field_gcd_mutation_x field_gcd_mutation_y")
    left = sp.Poly(x*x*y, x, y, modulus=5)
    right = sp.Poly(x*y*y, x, y, modulus=5)
    original = _native.prime_field_multivariate_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(left if side == "left" else right, field, replacement)
        return result

    monkeypatch.setattr(_native, "prime_field_multivariate_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(left, operation)(right)


@pytest.mark.parametrize("operation", ["gcd", "lcm"])
def test_lifting_retains_source_storage_and_terminal_native_failures(monkeypatch, operation):
    x, y = sp.symbols("field_gcd_lifting_x field_gcd_lifting_y")
    left = sp.Poly(x*x*y, x, y, modulus=5)
    right = sp.Poly(x*y*y, x, y, modulus=5)
    original = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        right._gens = (y, x)
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(left, operation)(right)
    monkeypatch.setattr(_finitefield, "_lift_native_result", original)
    right._gens = (x, y)

    def fail(*args, **kwargs):
        raise ValueError("test verifier budget exhausted")

    monkeypatch.setattr(_native, "prime_field_multivariate_native", fail)
    with pytest.raises(ValueError, match="verifier budget exhausted"):
        getattr(left, operation)(right)


def test_native_operation_admission_and_auxiliary_bound_are_explicit():
    x, y = sp.symbols("field_gcd_admission_x field_gcd_admission_y")
    sources = [_native_expr(x*y), _native_expr(x*x)]
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    for operation in ("gcd", "lcm"):
        for bad in [sources[:1], sources + sources[:1]]:
            with pytest.raises(ValueError):
                _native.prime_field_multivariate_native(bad, bindings, 5, operation)
        with pytest.raises(ValueError):
            _native.prime_field_multivariate_native(sources, bindings, 5, operation, order="grlex")
        with pytest.raises(ValueError):
            _native.prime_field_multivariate_native(sources, bindings, 5, operation, derivative_orders=[0, 0])
    variables = sp.symbols("field_gcd_auxiliary_0:8")
    left = sp.Poly(variables[0], *variables, modulus=5)
    right = sp.Poly(variables[-1], *variables, modulus=5)
    with pytest.raises(ValueError, match="auxiliary"):
        left.gcd(right)
    with pytest.raises(ValueError, match="auxiliary"):
        left.lcm(right)
