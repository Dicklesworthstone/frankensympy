"""Public multivariate prime-field arithmetic through the checked extension.

The coefficient and term-order fixtures were evaluated independently with
SymPy 1.14.0. Mutation tests assert whole-request binding, never replace the
native arithmetic with a Python implementation.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import _finitefield
from sympy.polys.polyerrors import (
    CoercionFailed, DomainError, ExactQuotientFailed, GeneratorsError,
    GeneratorsNeeded, PolynomialError, UnificationFailed,
)


CASES = [
    dict(p=2,
         add=[((2, 1), 1), ((0, 2), 1)],
         sub=[((2, 1), 1), ((0, 2), 1)],
         mul=[((2, 3), 1), ((2, 1), 1), ((0, 2), 1), ((0, 0), 1)],
         neg=[((2, 1), 1), ((0, 0), 1)],
         power=[((4, 2), 1), ((0, 0), 1)],
         monic=[((2, 1), 1), ((0, 0), 1)],
         diff=[((0, 0), 0)],
         divrem=[[((0, 0), 0)], [((2, 1), 1), ((0, 0), 1)]]),
    dict(p=3,
         add=[((1, 2), -1), ((1, 1), -1), ((0, 1), 1)],
         sub=[((1, 2), -1), ((1, 1), 1), ((0, 1), 1), ((0, 0), 1)],
         mul=[((2, 3), 1), ((1, 2), 1), ((1, 1), 1), ((0, 1), 1), ((0, 0), -1)],
         neg=[((1, 2), 1), ((0, 1), -1), ((0, 0), 1)],
         power=[((2, 4), 1), ((1, 3), 1), ((1, 2), -1), ((0, 2), 1), ((0, 1), 1), ((0, 0), 1)],
         monic=[((1, 2), 1), ((0, 1), -1), ((0, 0), 1)],
         diff=[((0, 1), 1)],
         divrem=[[((0, 1), 1)], [((0, 0), -1)]]),
    dict(p=5,
         add=[((2, 1), -2), ((1, 2), 2), ((1, 1), 2), ((0, 2), -2), ((0, 1), -1), ((0, 0), 1)],
         sub=[((2, 1), -2), ((1, 2), 2), ((1, 1), -2), ((0, 2), 2), ((0, 1), -1), ((0, 0), -1)],
         mul=[((3, 2), 1), ((2, 3), -2), ((2, 1), -2), ((1, 4), 1), ((0, 3), 2), ((0, 1), -1)],
         neg=[((2, 1), 2), ((1, 2), -2), ((0, 1), 1)],
         power=[((4, 2), -1), ((3, 3), 2), ((2, 4), -1), ((2, 2), -1), ((1, 3), 1), ((0, 2), 1)],
         monic=[((2, 1), 1), ((1, 2), -1), ((0, 1), -2)],
         diff=[((1, 0), 1), ((0, 1), -1)],
         divrem=[[((1, 0), -1)], [((1, 0), 1), ((0, 1), -1)]]),
    dict(p=7,
         add=[((2, 1), 3), ((1, 2), 2), ((1, 1), 2), ((0, 2), 3), ((0, 1), -3), ((0, 0), -1)],
         sub=[((2, 1), 3), ((1, 2), 2), ((1, 1), -2), ((0, 2), -3), ((0, 1), -3), ((0, 0), -3)],
         mul=[((3, 2), -1), ((2, 3), -1), ((2, 1), 3), ((1, 4), -1), ((1, 2), 3),
              ((1, 1), 3), ((0, 3), -2), ((0, 2), 1), ((0, 1), -3), ((0, 0), -2)],
         neg=[((2, 1), -3), ((1, 2), -2), ((0, 1), 3), ((0, 0), 2)],
         power=[((4, 2), 2), ((3, 3), -2), ((2, 4), -3), ((2, 2), 3), ((2, 1), 2),
                ((1, 3), 2), ((1, 2), -1), ((0, 2), 2), ((0, 1), -2), ((0, 0), -3)],
         monic=[((2, 1), 1), ((1, 2), 3), ((0, 1), -1), ((0, 0), -3)],
         diff=[((1, 0), -1), ((0, 1), -3)],
         divrem=[[((1, 0), -2)], [((1, 2), 1), ((1, 0), 2), ((0, 1), -3), ((0, 0), -2)]]),
]


@pytest.mark.parametrize("case", CASES, ids=["GF2", "GF3", "GF5", "GF7"])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "neg", "power", "monic", "diff", "divrem"])
def test_multivariate_operations_match_pinned_coefficient_fixtures(case, operation):
    x, y = sp.symbols("x y")
    left = sp.Poly(3*x**2*y + 2*x*y**2 + 4*y + 5, x, y, modulus=case["p"])
    right = sp.Poly(2*x*y + 3*y**2 + 1, x, y, modulus=case["p"])
    if operation == "power":
        result = (left**2,)
    elif operation == "diff":
        result = (left.diff(x, y),)
    elif operation == "divrem":
        result = left.div(right)
    elif operation in ("monic", "neg"):
        result = (getattr(left, operation)(),)
    else:
        result = (getattr(left, operation)(right),)
    expected = case[operation] if operation == "divrem" else [case[operation]]
    assert [value.terms() for value in result] == expected
    for value in result:
        assert value.gens == (x, y)
        assert value.gens[0] is x and value.gens[1] is y
        assert value.get_modulus() == case["p"]
    if operation == "divrem":
        assert result[0] * right + result[1] == left


@pytest.mark.parametrize("case", [
    "GF7_xy", "GF7_yx", "GF5_blocked", "GF5_three_blocked",
    "GF5_three_exact", "GF5_immediate",
])
def test_recursive_poly_division_and_sparse_reduction_have_distinct_pinned_semantics(case):
    x, y, z = sp.symbols("x y z")
    original = 3*x**2*y + 2*x*y**2 + 4*y + 5
    original_divisor = 2*x*y + 3*y**2 + 1
    three_divisor = (y*z + 1)*x + 1
    source, divisor, gens, prime, expected_q, expected_r, sparse_q, sparse_r = {
        "GF7_xy": (original, original_divisor, (x, y), 7,
                   -2*x, x*y**2 + 2*x - 3*y - 2,
                   -2*x - 3*y, 2*x + 2*y**3 - 2),
        "GF7_yx": (original, original_divisor, (y, x), 7,
                   3*x, -3*x**2*y - 3*x - 3*y - 2,
                   3*x, -3*x**2*y - 3*x - 3*y - 2),
        "GF5_blocked": (y*x**2 + (y + 2)*x + 1, y*x + 1, (x, y), 5,
                        x, x*y + x + 1, x + 1, x),
        "GF5_three_blocked": ((y*z + 1)*x**2 + (y*z**2 + 1)*x + z,
                              three_divisor, (x, y, z), 5,
                              x, x*y*z**2 + z, x + z, -x*z),
        "GF5_three_exact": ((x + z + 2)*three_divisor + y,
                            three_divisor, (x, y, z), 5,
                            x + z + 2, y, x + z + 2, y),
        "GF5_immediate": (sp.S.One, y, (x, y), 5,
                          sp.S.Zero, sp.S.One, sp.S.Zero, sp.S.One),
    }[case]
    left = sp.Poly(source, *gens, modulus=prime)
    right = sp.Poly(divisor, *gens, modulus=prime)
    quotient, remainder = left.div(right)
    assert quotient.as_expr() == expected_q and remainder.as_expr() == expected_r
    assert quotient * right + remainder == left
    assert all(a is b for a, b in zip(quotient.gens, gens))
    assert all(a is b for a, b in zip(remainder.gens, gens))
    sparse_quotients, sparse_remainder = sp.reduced(left, [right], order="lex")
    assert [q.as_expr() for q in sparse_quotients] == [sparse_q]
    assert sparse_remainder.as_expr() == sparse_r
    assert sparse_quotients[0] * right + sparse_remainder == left


def test_recursive_division_helpers_preserve_blocked_and_exact_coefficients():
    x, y = sp.symbols("x y")
    source = y*x**2 + (y + 2)*x + 1
    divisor = y*x + 1
    left = sp.Poly(source, x, y, modulus=5)
    right = sp.Poly(divisor, x, y, modulus=5)
    expected_q = sp.Poly(x, x, y, modulus=5)
    expected_r = sp.Poly(x*y + x + 1, x, y, modulus=5)
    assert left.quo(right) == expected_q
    assert left.rem(right) == expected_r
    assert sp.div(source, divisor, x, y, modulus=5) == (x, x*y + x + 1)
    assert sp.quo(source, divisor, x, y, modulus=5) == x
    assert sp.rem(source, divisor, x, y, modulus=5) == x*y + x + 1
    with pytest.raises(ExactQuotientFailed):
        left.exquo(right)
    exact_quotient = sp.Poly(x + y + 1, x, y, modulus=5)
    assert (exact_quotient * right).exquo(right) == exact_quotient


def test_recursive_division_zero_and_constant_boundaries_keep_the_declared_ring():
    x, y, z = sp.symbols("x y z")
    for gens in ((x, y), (y, x), (x, y, z)):
        zero = sp.Poly(0, *gens, modulus=5)
        source = sp.Poly(x + y + 1, *gens, modulus=5)
        assert zero.div(source) == (zero, zero)
        quotient, remainder = source.div(sp.Poly(2, *gens, modulus=5))
        assert quotient == sp.Poly(3*x + 3*y + 3, *gens, modulus=5)
        assert remainder == zero and quotient.gens == gens
        with pytest.raises(ZeroDivisionError):
            source.div(zero)
        with pytest.raises(ZeroDivisionError):
            zero.div(zero)


def test_constructor_inference_conversion_and_zero_coefficient_identity():
    x = sp.Symbol("x", positive=True)
    y = sp.Dummy("y")
    source = 7*x**2*y + 4*y**2 + 10
    explicit = sp.Poly(source, x, y, modulus=5)
    inferred = sp.Poly(source, domain="GF(5)")
    assert explicit.terms() == [((2, 1), 2), ((0, 2), -1)]
    assert set(inferred.gens) == {x, y}
    assert all(any(g is original for original in (x, y)) for g in inferred.gens)
    assert sp.Poly(explicit).gens == explicit.gens
    assert sp.Poly.from_poly(explicit).get_modulus() == 5
    assert sp.Poly(explicit, modulus=3).terms() == [((2, 1), -1), ((0, 2), -1)]
    assert sp.Poly(5*x**4*y + y, x, y, modulus=5).degree(x) == 0
    assert sp.Poly(3, x, y, modulus=5).terms() == [((0, 0), -2)]
    assert sp.Poly(0, x, y, modulus=5).terms() == [((0, 0), 0)]


def test_field_queries_project_complete_sparse_coordinates():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(3*x*y + 2*y**3 + 4, x, y, modulus=5)
    assert polynomial.coeffs() == [-2, 2, -1]
    assert polynomial.monoms() == [(1, 1), (0, 3), (0, 0)]
    assert polynomial.as_dict() == {(1, 1): -2, (0, 3): 2, (0, 0): -1}
    assert polynomial.degree() == 1 and polynomial.degree(y) == 3
    assert polynomial.degree(-1) == 3 and polynomial.degree_list() == (1, 3)
    assert polynomial.total_degree() == 3
    assert polynomial.LC() == -2 and polynomial.TC() == -1 and polynomial.EC() == -1
    assert polynomial.nth(0, 3) == 2 and polynomial.nth(1, 1) == -2
    assert polynomial.nth(8, 0) == 0
    assert sp.LC(polynomial) == -2 and sp.LM(polynomial) == x*y
    assert not polynomial.is_one and not polynomial.is_zero
    assert not polynomial.is_linear and not polynomial.is_quadratic
    assert polynomial.content() == 1 and polynomial.primitive() == (1, polynomial)
    assert sp.Poly(x, x, y, modulus=5).is_one is False
    assert sp.Poly(x + 1, x, y, modulus=5).is_linear
    assert sp.Poly(x*y, x, y, modulus=5).is_quadratic
    assert sp.Poly(x*y, x, y, modulus=5).TC() == 0
    zero = sp.Poly(0, x, y, modulus=5)
    assert zero.is_zero and zero.coeffs() == [0] and zero.content() == 0
    assert zero.total_degree() == 0 and zero.degree_list() == (-sp.oo, -sp.oo)
    assert sp.Poly(1, x, y, modulus=5).is_one
    for variables in ((x,), (x, y)):
        for value in (0, 1):
            constant = sp.Poly(value, *variables, modulus=5)
            assert constant.total_degree() == 0
            assert constant.is_linear and constant.is_quadratic
    with pytest.raises(PolynomialError, match="multivariate"):
        polynomial.all_coeffs()
    with pytest.raises(GeneratorsError):
        polynomial.nth(1)
    with pytest.raises(IndexError):
        polynomial.nth(0, -1)
    with pytest.raises(GeneratorsError):
        polynomial.degree(2)


@pytest.mark.parametrize("order,expected", [
    ("lex", [((2, 0, 1), 2), ((2, 0, 0), -3), ((1, 2, 0), 3), ((0, 4, 0), 1), ((0, 0, 2), 1)]),
    ("grlex", [((0, 4, 0), 1), ((2, 0, 1), 2), ((1, 2, 0), 3), ((2, 0, 0), -3), ((0, 0, 2), 1)]),
    ("grevlex", [((0, 4, 0), 1), ((1, 2, 0), 3), ((2, 0, 1), 2), ((2, 0, 0), -3), ((0, 0, 2), 1)]),
])
def test_three_supported_term_orders_match_pinned_oracle(order, expected):
    x, y, z = sp.symbols("x y z")
    polynomial = sp.Poly(2*x**2*z + 3*x*y**2 + 4*x**2 + y**4 + z**2,
                         x, y, z, modulus=7)
    assert polynomial.terms(order=order) == expected
    assert polynomial.coeffs(order=order) == [c for _, c in expected]
    assert polynomial.monoms(order=order) == [m for m, _ in expected]


def test_mixed_partial_specs_use_one_cumulative_native_request(monkeypatch):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x**3*y**4 + 2*x*y**2 + 3, x, y, modulus=5)
    original = _native.prime_field_multivariate_native
    calls = []

    def observed(*args, **kwargs):
        calls.append((args[3], args[5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_multivariate_native", observed)
    result = polynomial.diff((x, 2), (y, 1))
    assert calls == [("diff", [2, 1])]
    assert result.terms() == [((1, 3), -1)]
    assert polynomial.diff(0, 0, 1) == result
    assert polynomial.diff((x, 0), (y, 0)) == polynomial
    assert sp.Poly(x**5*y**5, x, y, modulus=5).diff(x, y).is_zero
    with pytest.raises(ValueError, match="0..64"):
        polynomial.diff((x, 33), (y, 32))


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_ordered_reduction_has_all_quotient_coordinates_and_irreducible_remainder(order):
    x, y = sp.symbols("x y")
    source = x*y**2 + x**2*y + y**3 + 1
    divisors = [x*y - 1, y**2 - x]
    quotients, remainder = sp.reduced(source, divisors, x, y, modulus=5, order=order)
    expected_q = [x + y, -1] if order == "lex" else [x + y + 1, y]
    expected_r = y**3 + y**2 + y + 1 if order == "lex" else x + y + 2
    assert quotients == expected_q and remainder == expected_r
    poly = sp.Poly(source, x, y, modulus=5)
    basis = [sp.Poly(value, x, y, modulus=5) for value in divisors]
    qs, rem = sp.reduced(poly, basis, order=order)
    assert all(type(value) is sp.Poly for value in [*qs, rem])
    assert sum((q*g for q, g in zip(qs, basis)), sp.Poly(0, x, y, modulus=5)) + rem == poly
    assert sp.reduced(poly, basis, order=order, polys=False) == (expected_q, expected_r)
    assert sp.reduced(source, divisors, x, y, modulus=5, order=order, polys=True) == (qs, rem)


def test_reduction_empty_repeated_constant_and_zero_divisors():
    from sympy.polys import reduced as polys_reduced
    from sympy.polys import polytools

    assert sp.reduced is polys_reduced is polytools.reduced
    assert "reduced" in polytools.__all__
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 2, x, y, modulus=5)
    divisor = sp.Poly(x, x, y, modulus=5)
    assert sp.reduced(polynomial, []) == ([], polynomial)
    qs, rem = sp.reduced(polynomial, [divisor, divisor])
    assert qs == [sp.Poly(y, x, y, modulus=5), sp.Poly(0, x, y, modulus=5)]
    assert rem == sp.Poly(2, x, y, modulus=5)
    quotient, remainder = polynomial.div(2)
    assert quotient.terms() == [((1, 1), -2), ((0, 0), 1)] and remainder.is_zero
    assert (polynomial * divisor).exquo(divisor) == polynomial
    with pytest.raises(ExactQuotientFailed):
        polynomial.exquo(divisor)
    with pytest.raises(ZeroDivisionError):
        polynomial.div(sp.Poly(0, x, y, modulus=5))
    with pytest.raises(ZeroDivisionError):
        sp.reduced(polynomial, [divisor, sp.Poly(0, x, y, modulus=5)])
    with pytest.raises(NotImplementedError, match="prime-field"):
        sp.reduced(x*y, [x], x, y)
    with pytest.raises(GeneratorsNeeded):
        sp.reduced(x*y, [x, y], modulus=5)


def test_reduction_explicit_field_conversion_is_bound_to_requested_domain():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 2, x, y, modulus=5)
    for divisor in (sp.Poly(x, x, y, modulus=7), sp.Poly(x, x, y)):
        with pytest.raises(UnificationFailed):
            sp.reduced(polynomial, [divisor])
        expected = ([sp.Poly(y, x, y, modulus=5)], sp.Poly(2, x, y, modulus=5))
        assert sp.reduced(polynomial, [divisor], x, y, modulus=5) == expected
        assert sp.reduced(polynomial, [divisor], x, y, domain="GF(5)") == expected


def test_scalar_arithmetic_signed_lifts_and_module_division():
    x, y = sp.symbols("x y")
    signed = sp.Poly(4*x*y + 3, x, y, modulus=5)
    unsigned = sp.Poly(4*x*y + 3, x, y, modulus=5, symmetric=False)
    assert signed.terms() == [((1, 1), -1), ((0, 0), -2)]
    assert unsigned.terms() == [((1, 1), 4), ((0, 0), 3)]
    assert signed == unsigned and hash(signed) == hash(unsigned)
    assert (2 - unsigned).terms() == [((1, 1), 1), ((0, 0), 4)]
    assert (unsigned * 2).terms() == [((1, 1), 3), ((0, 0), 1)]
    assert (unsigned + 3).terms() == [((1, 1), 4), ((0, 0), 1)]
    assert unsigned.monic().terms() == [((1, 1), 1), ((0, 0), 2)]
    assert (+unsigned).terms() == unsigned.terms()
    assert sp.div(x*y + x, x, x, y, modulus=5) == (y + 1, 0)
    assert sp.rem(x*y + 2, x, x, y, modulus=5) == 2
    assert sp.quo(x*y + 2, x, x, y, modulus=5) == y


def test_pure_polynomial_names_do_not_replace_exponent_coordinates():
    x, y, u, v = sp.symbols("x y u v")
    left = sp.PurePoly(x**2*y + 4, x, y, modulus=5)
    right = sp.Poly(u**2*v + 4, u, v, modulus=5)
    wrong = sp.PurePoly(u*v**2 + 4, u, v, modulus=5)
    assert left == right and right == left and hash(left) == hash(right)
    assert left != wrong
    assert left != sp.Poly(x**2 + 4, x, modulus=5)
    assert sp.Poly(x + y, x, y, modulus=5) == x + y
    assert sp.Poly(x + y, x, y, modulus=5) != x + u


def test_typed_symbols_assumptions_and_all_eight_generators_survive():
    positive = sp.Symbol("field_multi_same", positive=True)
    negative = sp.Symbol("field_multi_same", negative=True)
    dummy = sp.Dummy("field_multi_same")
    variables = (positive, negative, dummy, *sp.symbols("a b c d e"))
    polynomial = sp.Poly(sum(variables), *variables, modulus=5)
    result = polynomial**2
    assert all(a is b for a, b in zip(result.gens, variables))
    assert result.degree_list() == (2,) * 8
    assert sp.Poly(positive + negative, positive, negative, modulus=3).gens == (positive, negative)
    with pytest.raises((ValueError, GeneratorsError)):
        sp.Poly(positive, positive, positive, modulus=5)
    with pytest.raises(GeneratorsError):
        sp.Poly(0, *variables, sp.Symbol("ninth"), modulus=5)
    with pytest.raises((ValueError, GeneratorsError)):
        sp.Poly(sum((*variables, sp.Symbol("ninth"))), modulus=5)


def test_noncommutative_declarations_never_enter_a_commutative_field_ring(monkeypatch):
    x = sp.Symbol("noncommutative_field_x", commutative=False)
    y = sp.Symbol("y")

    def forbidden(*args, **kwargs):
        raise AssertionError("a noncommutative source reached prime-field arithmetic")

    monkeypatch.setattr(_native, "prime_field_multivariate_native", forbidden)
    monkeypatch.setattr(_native, "prime_field_polynomial_native", forbidden)
    for expression in (x, y, sp.Mul(0, x, evaluate=False)):
        with pytest.raises(GeneratorsError, match="commutative"):
            sp.Poly(expression, x, y, modulus=5)
    with pytest.raises(GeneratorsError, match="commutative"):
        sp.Poly(x, x, modulus=5)


@pytest.mark.parametrize("operation", [
    lambda p, q: p.gcd(q), lambda p, q: p.lcm(q), lambda p, q: p.gcdex(q),
    lambda p, q: p.invert(q), lambda p, q: p.factor_list(), lambda p, q: p.sqf_list(),
    lambda p, q: p.resultant(q), lambda p, q: p.discriminant(),
    lambda p, q: p.compose(q), lambda p, q: p.shift(2),
])
def test_unsupported_multivariate_families_refuse_before_univariate_arithmetic(monkeypatch, operation):
    x, y = sp.symbols("x y")
    left = sp.Poly(x*y + 1, x, y, modulus=5)
    right = sp.Poly(x + y, x, y, modulus=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("multivariate input reached a univariate or rational-domain kernel")

    for name in ("prime_field_polynomial_native", "prime_field_polynomial_factor_native",
                 "poly_multivariate_gcd_native", "poly_multivariate_div_rem_native",
                 "poly_multivariate_lcm_native", "poly_factor_multivariate_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(NotImplementedError):
        operation(left, right)


def test_mixed_domains_gens_and_unsupported_options_are_terminal():
    x, y, z = sp.symbols("x y z")
    left = sp.Poly(x*y + 1, x, y, modulus=5)
    for right in [sp.Poly(x*y + 1, x, y), sp.Poly(x*y + 1, x, y, modulus=7),
                  sp.Poly(x*z + 1, x, z, modulus=5), sp.Poly(x*y + 1, y, x, modulus=5), x + y]:
        with pytest.raises(UnificationFailed):
            left + right
    with pytest.raises(DomainError):
        sp.Poly(x*y, x, y, modulus=5, domain="GF(7)")
    with pytest.raises(NotImplementedError):
        sp.Poly.from_list([1, 2], (x, y), modulus=5)
    with pytest.raises(NotImplementedError):
        sp.Poly(x*y, x, y, modulus=5, extension=True)
    with pytest.raises(ValueError):
        sp.reduced(left, [], order="ilex")
    with pytest.raises(ValueError):
        left.terms(order="ilex")
    with pytest.raises(TypeError):
        sp.reduced(left, (value for value in [left]))
    with pytest.raises(ValueError, match="32"):
        sp.reduced(left, [left] * 33)
    with pytest.raises(TypeError):
        sp.reduced(left, [], polys=1)
    with pytest.raises(GeneratorsError):
        sp.factor_list(sp.Poly(x + 1, x, modulus=5), x, y)


@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power", "cancellation"])
def test_original_unsupported_children_do_not_disappear(wrapper):
    x, y = sp.symbols("x y")
    invalid = sp.sin(y)
    if wrapper == "zero_product":
        expression = sp.Mul(0, invalid, evaluate=False)
    elif wrapper == "zero_power":
        expression = sp.Pow(invalid, 0, evaluate=False)
    else:
        expression = sp.Add(invalid, sp.Mul(-1, invalid, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        sp.Poly(sp.Add(x, expression, evaluate=False), x, y, modulus=5)


def test_original_coefficient_degree_and_source_admission_bounds():
    x, y = sp.symbols("x y")
    for expression in (x*y + sp.Rational(1, 2), x*y + sp.Rational(1, 5)):
        with pytest.raises(CoercionFailed):
            sp.Poly(expression, x, y, modulus=5)
    half_sum = sp.Add(sp.Rational(1, 2), sp.Rational(1, 2), evaluate=False)
    assert sp.Poly(sp.Mul(half_sum, x, y, evaluate=False), x, y, modulus=5).terms() == [((1, 1), 1)]
    for expression in (x**33*y**32, 5*x**33*y**32, sp.Integer(1 << 4097)*x*y):
        with pytest.raises(ValueError):
            sp.Poly(expression, x, y, modulus=5)
    with pytest.raises(ValueError):
        sp.Poly(x**32*y**32, x, y, modulus=5)**2
    assert sp.Poly(x**32*y**32, x, y, modulus=5).total_degree() == 64


@pytest.mark.parametrize("field,replacement", [
    ("_modulus", 7), ("_domain", "GF(7)"), ("_symmetric", False),
])
def test_operand_metadata_mutation_after_native_work_prevents_publication(monkeypatch, field, replacement):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, modulus=5)
    original = _native.prime_field_multivariate_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(polynomial, field, replacement)
        return result

    monkeypatch.setattr(_native, "prime_field_multivariate_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial**2


def test_ordered_divisor_container_is_bound_through_scalar_admission(monkeypatch):
    x, y = sp.symbols("x y")
    source = x*y + 2
    divisors = [x]
    modulus = sp.Integer(5)
    original = _native.prime_field_source_preflight_native

    def mutate(*args, **kwargs):
        divisors[0] = y
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.reduced(source, divisors, x, y, modulus=modulus)


def test_divisor_and_source_generators_remain_bound_during_output_lifting(monkeypatch):
    x, y = sp.symbols("x y")
    source = sp.Poly(x*y + 2, x, y, modulus=5)
    divisor = sp.Poly(x, x, y, modulus=5)
    divisors = [divisor]
    original = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        divisors.append(divisor)
        source._gens = (y, x)
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.reduced(source, divisors)


def test_projection_and_equality_retain_both_original_sources(monkeypatch):
    x, y = sp.symbols("x y")
    left = sp.Poly(x*y + 1, x, y, modulus=5)
    right = sp.Poly(x*y + 1, x, y, modulus=5)
    original = _native.prime_field_multivariate_terms_native
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            left._modulus = 7
        return result

    monkeypatch.setattr(_native, "prime_field_multivariate_terms_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        left == right


def test_held_source_and_stale_public_assumptions_are_not_reinterpreted():
    x = sp.Symbol("multivariate_stale", positive=True)
    y = sp.Symbol("y")
    source = sp.Add(x, y, evaluate=False)
    source._args = (x, x)
    with pytest.raises(ValueError):
        sp.Poly(source, x, y, modulus=5)
    original = dict(x._assumptions)
    try:
        x._assumptions["positive"] = False
        with pytest.raises(ValueError, match="identity"):
            sp.Poly(x*y, x, y, modulus=5)
    finally:
        x._assumptions.clear()
        x._assumptions.update(original)


def test_only_exact_classes_and_source_storage_are_admitted():
    x, y = sp.symbols("x y")

    class CustomPoly(sp.Poly):
        pass

    with pytest.raises(TypeError):
        CustomPoly(x*y, x, y, modulus=5)
    polynomial = sp.Poly(x*y, x, y, modulus=5)
    polynomial.as_expr = lambda: sp.Integer(0)
    with pytest.raises(TypeError, match="as_expr"):
        polynomial**2


def test_hostile_metaclass_cannot_impersonate_an_admitted_builtin():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, modulus=5)
    calls = []

    class HostileMeta(type):
        def __eq__(cls, other):
            calls.append("metaclass equality")
            raise AssertionError("admission invoked an untrusted metaclass")

        __hash__ = type.__hash__

    class Hostile(metaclass=HostileMeta):
        pass

    class HostileList(list, metaclass=HostileMeta):
        pass

    class HostilePoly(sp.Poly, metaclass=HostileMeta):
        pass

    value = Hostile()
    assert not _finitefield.is_finite(value)
    assert polynomial != value
    with pytest.raises(GeneratorsError):
        sp.Poly(x*y, x, value, modulus=5)
    with pytest.raises(CoercionFailed):
        sp.Poly(x*y, x, y, modulus=value)
    with pytest.raises(UnificationFailed):
        polynomial + value
    with pytest.raises(CoercionFailed):
        polynomial**value
    with pytest.raises(CoercionFailed):
        polynomial.diff((x, value))
    with pytest.raises(CoercionFailed):
        polynomial.nth(value, 0)
    with pytest.raises(TypeError):
        sp.reduced(polynomial, HostileList([polynomial]))
    with pytest.raises(TypeError):
        HostilePoly(x*y, x, y, modulus=5)
    assert calls == []


@pytest.mark.parametrize("operation", ["pow", "reduced", "terms"])
def test_budget_and_verifier_failures_do_not_restart_in_another_domain(monkeypatch, operation):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise ValueError("independent multivariate verifier budget exhausted")

    name = ("prime_field_multivariate_terms_native" if operation == "terms" else
            "prime_field_multivariate_native")
    monkeypatch.setattr(_native, name, refuse)
    with pytest.raises(ValueError, match="verifier budget"):
        if operation == "pow":
            polynomial**0
        elif operation == "reduced":
            sp.reduced(polynomial, [])
        else:
            polynomial.coeffs()
    assert calls == [1]


@pytest.mark.parametrize("operation,count,orders,order", [
    ("add", 1, None, "lex"), ("normalize", 2, None, "lex"),
    ("reduced", 34, None, "lex"), ("gcd", 2, None, "lex"),
    ("diff", 1, [], "lex"), ("normalize", 1, [0, 0], "lex"),
    ("normalize", 1, None, "unknown"),
    ("divrem", 2, None, "grlex"), ("divrem", 2, None, "grevlex"),
])
def test_native_entry_rejects_malformed_operation_shapes(operation, count, orders, order):
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError):
        _native.prime_field_multivariate_native(
            [_native_expr(x*y)] * count, [_symbol_binding(x), _symbol_binding(y)],
            5, operation, 0, orders, order)


@pytest.mark.parametrize("shape", ["empty", "duplicate", "excess"])
def test_native_entry_rejects_invalid_generator_universes(shape):
    x = sp.Symbol("x")
    bindings = [] if shape == "empty" else [_symbol_binding(x)] * (2 if shape == "duplicate" else 9)
    with pytest.raises(ValueError):
        _native.prime_field_multivariate_native([_native_expr(x)], bindings, 5, "normalize")
