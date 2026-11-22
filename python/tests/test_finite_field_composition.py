"""Checked first-generator composition in the original prime-field ring.

Complete coefficient fixtures were obtained in an isolated SymPy 1.14.0
process. Mathematical acceptance remains the native coefficient verifier;
point evaluation alone cannot establish polynomial identity in finite fields.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import _finitefield
from sympy.polys.polyerrors import UnificationFailed


COEFFICIENTS = {
    2: [((0, 5), 1), ((0, 1), 1), ((0, 0), 1)],
    3: [((1, 3), 1), ((0, 2), -1), ((0, 1), 1), ((0, 0), -1)],
    5: [((2, 3), 2), ((1, 4), 1), ((1, 3), -1), ((1, 2), 2),
        ((0, 5), 2), ((0, 4), 1), ((0, 3), -2), ((0, 2), 2), ((0, 1), 2)],
    7: [((2, 3), -2), ((1, 4), 1), ((1, 3), -3), ((1, 2), -2),
        ((0, 5), -1), ((0, 4), -1), ((0, 3), -3), ((0, 2), 2), ((0, 0), -2)],
    65521: [((2, 3), 12), ((1, 4), 36), ((1, 3), 4), ((1, 2), 12),
            ((0, 5), 27), ((0, 4), 6), ((0, 3), 18), ((0, 2), 2),
            ((0, 1), 7), ((0, 0), 5)],
}


@pytest.mark.parametrize("prime", COEFFICIENTS)
@pytest.mark.parametrize("symmetric", [True, False])
def test_multivariate_composition_matches_pinned_complete_coefficients(prime, symmetric):
    x, y = sp.symbols("x y")
    left = sp.Poly(3*x**2*y + 2*x*y**2 + 4*y + 5, x, y,
                   modulus=prime, symmetric=symmetric)
    right = sp.Poly(2*x*y + 3*y**2 + 1, x, y,
                    modulus=prime, symmetric=symmetric)
    result = left.compose(right)
    expected = COEFFICIENTS[prime]
    if not symmetric:
        expected = [(monomial, coefficient % prime) for monomial, coefficient in expected]
    assert result.terms() == expected
    assert result.get_modulus() == prime
    assert result.gens[0] is x and result.gens[1] is y
    assert result._symmetric is symmetric


@pytest.mark.parametrize("reverse", [False, True])
def test_composition_uses_the_first_declared_coordinate(reverse):
    x, y = sp.symbols("x y")
    generators = (y, x) if reverse else (x, y)
    source = sp.Poly(x**2 + y, *generators, modulus=5)
    replacement = sp.Poly(x + y, *generators, modulus=5)
    expected = x**2 + x + y if reverse else x**2 + 2*x*y + y**2 + y
    result = source.compose(replacement)
    assert result == sp.Poly(expected, *generators, modulus=5)
    assert result.gens == generators


def test_module_composition_preserves_expression_and_poly_output_contracts():
    x, y, z = sp.symbols("x y z")
    source, replacement = x*x*y + z, x + y*z
    expected = x*x*y + 2*x*y*y*z + y**3*z*z + z
    assert sp.compose(source, replacement, x, y, z, modulus=5) == expected
    result = sp.compose(source, replacement, x, y, z, domain="GF(5)", polys=True)
    assert type(result) is sp.Poly and result.as_expr() == expected
    assert result.get_modulus() == 5 and result.gens == (x, y, z)
    polynomial = sp.Poly(source, x, y, z, modulus=5)
    assert sp.compose(polynomial, sp.Poly(replacement, x, y, z, modulus=5)) == result
    assert sp.compose(polynomial, sp.Poly(replacement, x, y, z, modulus=5), polys=False) == expected


def test_zero_constants_identity_and_unused_axes_remain_in_the_declared_ring():
    x, y, unused = sp.symbols("x y unused")
    variables = (x, y, unused)
    zero = sp.Poly(0, *variables, modulus=5)
    constant = sp.Poly(3, *variables, modulus=5)
    source = sp.Poly(x**3*y + 2*y*y, *variables, modulus=5)
    identity = sp.Poly(x, *variables, modulus=5)
    assert source.compose(identity) == source
    assert source.compose(zero) == sp.Poly(2*y*y, *variables, modulus=5)
    assert source.compose(3) == sp.Poly(2*y + 2*y*y, *variables, modulus=5)
    for left, right, expected in [(zero, source, zero), (constant, source, constant)]:
        result = left.compose(right)
        assert result == expected and result.gens == variables
        assert all(actual is original for actual, original in zip(result.gens, variables))
    inactive = sp.Poly(y*y + 3, *variables, modulus=5)
    assert inactive.compose(sp.Poly(x**64, *variables, modulus=5)) == inactive


def test_same_display_symbols_and_pure_polys_retain_typed_coordinates():
    positive = sp.Symbol("compose_same", positive=True)
    negative = sp.Symbol("compose_same", negative=True)
    dummy = sp.Dummy("compose_same")
    variables = (positive, negative, dummy)
    source = sp.PurePoly(positive**2*negative + dummy, *variables, modulus=5)
    replacement = sp.Poly(positive + negative*dummy, *variables, modulus=5)
    result = source.compose(replacement)
    expected = sp.PurePoly(positive**2*negative + 2*positive*negative**2*dummy
                           + negative**3*dummy**2 + dummy, *variables, modulus=5)
    assert type(result) is sp.PurePoly and result == expected
    assert all(actual is original for actual, original in zip(result.gens, variables))


def test_composition_is_a_ring_identity_not_a_function_table_over_field_points():
    x, y = sp.symbols("x y")
    result = sp.Poly(x*x - x, x, y, modulus=2).compose(sp.Poly(y, x, y, modulus=2))
    assert result.terms() == [((0, 2), 1), ((0, 1), 1)]
    assert not result.is_zero
    assert result.eval({x: 0, y: 0}) == result.eval({x: 1, y: 1}) == 0


def test_composition_uses_one_checked_multivariate_request_without_domain_retry(monkeypatch):
    x, y = sp.symbols("x y")
    source = sp.Poly(x**2 + y, x, y, modulus=5)
    replacement = sp.Poly(x + y, x, y, modulus=5)
    original = _native.prime_field_multivariate_native
    calls = []

    def checked(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("composition reached an unrelated arithmetic lane")

    monkeypatch.setattr(_native, "prime_field_multivariate_native", checked)
    monkeypatch.setattr(_native, "prime_field_polynomial_native", forbidden)
    monkeypatch.setattr(_native, "poly_compose_native", forbidden)
    result = source.compose(replacement)
    assert calls == ["compose"]
    assert result.gens == (x, y) and result.get_modulus() == 5


def test_constant_composition_cannot_bypass_field_or_generator_admission():
    x, y = sp.symbols("x y")
    alternate_x = sp.Symbol("x", positive=True)
    zero = sp.Poly(0, x, y, modulus=5)
    for wrong in (sp.Poly(x + y, x, y), sp.Poly(x + y, x, y, modulus=7),
                  sp.Poly(0, y, x, modulus=5), sp.Poly(0, alternate_x, y, modulus=5),
                  sp.Poly(0, x, modulus=5), sp.Rational(1, 2), sp.Float(0)):
        with pytest.raises(UnificationFailed):
            zero.compose(wrong)


@pytest.mark.parametrize("side", ["source", "replacement"])
@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power"])
def test_composition_admits_held_original_poles_even_when_the_output_would_be_constant(side, wrapper):
    x, y = sp.symbols("x y")
    source = sp.Poly(0, x, y, modulus=5)
    replacement = sp.Poly(0, x, y, modulus=5)
    pole = sp.Pow(x, -1, evaluate=False)
    held = (sp.Mul(0, pole, evaluate=False) if wrapper == "zero_product"
            else sp.Pow(pole, 0, evaluate=False))
    operand = source if side == "source" else replacement
    operand._expr = held
    with pytest.raises(ValueError):
        source.compose(replacement)


@pytest.mark.parametrize("side,field", [("source", "_symmetric"), ("replacement", "_gens")])
def test_operand_mutation_after_native_check_prevents_publication(monkeypatch, side, field):
    x, y = sp.symbols("x y")
    source = sp.Poly(x*x + y, x, y, modulus=5)
    replacement = sp.Poly(x + y, x, y, modulus=5)
    original = _native.prime_field_multivariate_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        operand = source if side == "source" else replacement
        setattr(operand, field, False if field == "_symmetric" else (y, x))
        return result

    monkeypatch.setattr(_native, "prime_field_multivariate_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.compose(replacement)


def test_replacement_mutation_during_lifting_prevents_publication(monkeypatch):
    x, y = sp.symbols("x y")
    source = sp.Poly(x*x + y, x, y, modulus=5)
    replacement = sp.Poly(x + y, x, y, modulus=5)
    original = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement._expr = sp.Integer(0)
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.compose(replacement)


@pytest.mark.parametrize("reason", ["verifier budget exhausted", "composition cancelled"])
def test_operational_failures_are_terminal_even_for_zero_composition(monkeypatch, reason):
    x, y = sp.symbols("x y")
    source = sp.Poly(0, x, y, modulus=5)
    replacement = sp.Poly(x + y, x, y, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args[3])
        raise ValueError(reason)

    monkeypatch.setattr(_native, "prime_field_multivariate_native", refuse)
    with pytest.raises(ValueError, match=reason):
        source.compose(replacement)
    assert calls == ["compose"]


def test_degree_growth_retains_coefficient_axes_and_refuses_before_expansion():
    x, y = sp.symbols("x y")
    replacement = sp.Poly(y*y, x, y, modulus=5)
    assert sp.Poly(x**32, x, y, modulus=5).compose(replacement) == sp.Poly(y**64, x, y, modulus=5)
    with pytest.raises(ValueError, match="composition degree"):
        sp.Poly(x**32*y, x, y, modulus=5).compose(replacement)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_native_composition_requires_exactly_two_original_sources(count):
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="operand count"):
        _native.prime_field_multivariate_native(
            [_native_expr(x + y)] * count, [_symbol_binding(x), _symbol_binding(y)],
            5, "compose",
        )
