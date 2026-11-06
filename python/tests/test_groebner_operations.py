"""Checked public Groebner operations; require the actual native extension."""

import pytest

from sympy import Poly, Pow, PurePoly, Rational, Symbol, expand, groebner


def test_fglm_all_order_pairs_agree_with_independent_basis_generation():
    x, y = Symbol("x"), Symbol("y")
    generators = [x * y - 1, y**2 - x]
    for source_order in ("lex", "grlex", "grevlex"):
        source = groebner(generators, x, y, order=source_order)
        assert source.is_zero_dimensional
        for target in ("lex", "grlex", "grevlex"):
            converted = source.fglm(target)
            assert converted.exprs == groebner(generators, x, y, order=target).exprs
            assert converted.order == target
            assert converted.gens == (x, y)
            if target == source_order:
                assert converted is source
    assert source.fglm("lex").exprs == [x - y**2, y**3 - 1]


def test_rational_domain_and_monic_normalization_survive_conversion():
    x, y = Symbol("x"), Symbol("y")
    basis = groebner([2*x - y, y**2 - 2], x, y, order="grevlex", domain="QQ")
    converted = basis.fglm("lex")
    assert str(converted.domain) == "QQ"
    assert converted.exprs == [x - y/2, y**2 - 2]
    inferred = groebner([x - y/2, y**2 - 2], x, y, order="grevlex")
    assert str(inferred.domain) == "QQ"
    assert inferred.fglm("lex").exprs == converted.exprs
    integer_basis = groebner([2*x - y, y**2 - 2], x, y, order="grevlex")
    assert str(integer_basis.domain) == "ZZ"
    assert integer_basis.fglm("lex").exprs == [2*x - y, y**2 - 2]


def test_reduction_quotients_reconstruct_input_with_original_basis_scaling():
    x, y = Symbol("x"), Symbol("y")
    for domain in ("ZZ", "QQ"):
        basis = groebner([2*x - y, y**2 - 2], x, y, order="grevlex", domain=domain)
        f = x**2 + 2*y
        quotients, remainder = basis.reduce(f)
        assert len(quotients) == len(basis)
        assert expand(f - sum(q*g for q, g in zip(quotients, basis)) - remainder) == 0
        assert remainder == 2*y + Rational(1, 2)
        assert basis.contains((x + y)*(2*x - y))
        assert not basis.contains(x)
        assert basis.contains(0)
    assert basis.reduce(x, auto=False)[1] == y/2


def test_positive_dimension_empty_and_unit_ideals_have_explicit_behavior():
    x, y = Symbol("x"), Symbol("y")
    for basis in (groebner([x*y], x, y), groebner([], x, y), groebner([1], x, y)):
        assert not basis.is_zero_dimensional
        assert basis.fglm("lex") is basis
        with pytest.raises(NotImplementedError):
            basis.fglm("grlex")
    assert groebner([], x, y).reduce(x + y) == ([], x + y)
    assert groebner([1], x, y).contains(x + y)


def test_mutating_basis_cannot_turn_incomplete_reduction_into_membership_claim():
    x, y = Symbol("x"), Symbol("y")
    basis = groebner([x, y], x, y)
    basis.exprs[:] = [x*y - 1, y**2 - x]
    # This generating set is not a Groebner basis in lex order: it omits
    # the normal forms needed for a complete ideal-membership decision.
    with pytest.raises(ValueError):
        basis.contains(x - y**2)
    with pytest.raises(ValueError):
        _ = basis.is_zero_dimensional


def test_assumption_bearing_same_spelling_generators_retain_identity():
    plain = Symbol("x")
    positive = Symbol("x", positive=True)
    basis = groebner([positive - plain**2, plain**3 - 1], positive, plain, order="grevlex")
    converted = basis.fglm("lex")
    assert converted.gens == (positive, plain)
    assert converted.contains(positive - plain**2)
    assert not converted.contains(positive - plain)


def test_unsupported_domains_and_integer_ring_reduction_do_not_silently_coerce():
    x = Symbol("x")
    with pytest.raises(NotImplementedError):
        groebner([x - 1], x, domain="RR")
    with pytest.raises(ValueError):
        groebner([x - Rational(1, 2)], x, domain="ZZ")
    with pytest.raises(NotImplementedError):
        groebner([2*x - 1], x).reduce(x, auto=False)
    with pytest.raises(ValueError):
        groebner([x - 1], x).reduce(x/2)
    with pytest.raises(ValueError):
        groebner([x], x).fglm("unrecognized")


def test_poly_input_and_original_expression_bounds():
    x, y = Symbol("x"), Symbol("y")
    basis = groebner([x**2 - 1], x, domain="QQ")
    assert basis.reduce(Poly(x**3, x)) == ([x], x)
    with pytest.raises(ValueError):
        basis.reduce(Poly(y, y))
    for held in (Pow(2, 10**9, evaluate=False), Pow(x, 10**9, evaluate=False)):
        with pytest.raises(ValueError):
            basis.reduce(held)


@pytest.mark.parametrize("options", [
    {"modulus": 2}, {"modulus": 5}, {"modulus": 0},
    {"domain": "GF(2)"}, {"domain": "RR"}, {"domain": "QQ", "modulus": 2},
    {"extension": True}, {"field": True}, {"polys": True}, {"method": "f5b"},
])
def test_unsupported_ring_and_output_options_are_refused_before_native_work(monkeypatch, options):
    from sympy.polys import polytools

    x = Symbol("domain_guard_x")
    sources = [x**2 + 1, x + 1]

    def forbidden(*args):
        raise AssertionError("unsupported Groebner request reached the rational engine")

    monkeypatch.setattr(polytools._native, "groebner_basis_native", forbidden)
    with pytest.raises(NotImplementedError):
        groebner(sources, x, **options)


def test_characteristic_two_counterexample_cannot_return_the_rational_unit_ideal():
    x = Symbol("characteristic_two_x")
    # In QQ, (x**2+1) - (x-1)*(x+1) = 2, so the ideal is the whole ring.
    # In GF(2), x**2+1 = (x+1)**2 and the ideal is the proper ideal <x+1>.
    # A QQ result is therefore a mathematically false answer to this request.
    assert groebner([x**2 + 1, x + 1], x, domain="QQ").exprs == [1]
    with pytest.raises(NotImplementedError, match="finite-field"):
        groebner([x**2 + 1, x + 1], x, modulus=2)


@pytest.mark.parametrize("poly_type", [Poly, PurePoly])
def test_finite_poly_domains_cannot_be_erased_by_basis_and_ideal_operations(poly_type):
    x = Symbol("retained_field_x")
    finite = poly_type(x**2 + 1, x, modulus=2)
    basis = groebner([x + 1], x, domain="QQ")
    for operation in (
        lambda: groebner([finite, x + 1], x),
        lambda: groebner([finite], x, domain="QQ"),
        lambda: basis.reduce(finite),
        lambda: basis.contains(finite),
        lambda: basis.invert(finite),
        lambda: basis.multiplication_matrix(finite),
    ):
        with pytest.raises(NotImplementedError, match="ZZ and QQ Poly"):
            operation()
    # Directly replacing the public basis list must not bypass domain admission.
    basis.exprs[:] = [finite]
    with pytest.raises(NotImplementedError):
        _ = basis.is_zero_dimensional
    with pytest.raises(NotImplementedError):
        basis.eliminate(x)


@pytest.mark.parametrize("poly_type", [Poly, PurePoly])
def test_rational_poly_inputs_preserve_explicit_domain_without_public_unwrapping(poly_type):
    x = Symbol("rational_poly_input_x")
    polynomial = poly_type(2*x + 1, x, domain="QQ")
    basis = groebner([polynomial], x, domain="QQ")
    assert basis.domain == "QQ"
    assert basis.exprs == [x + Rational(1, 2)]
    assert groebner([polynomial], x).domain == "ZZ"
    assert basis.reduce(poly_type(x**2, x, domain="QQ"))[1] == Rational(1, 4)
    assert groebner([x - 1], x, modulus=None, domain=None).exprs == [x - 1]

    def forbidden():
        raise AssertionError("an instance override was invoked during exact source admission")

    polynomial.as_expr = forbidden
    with pytest.raises(TypeError, match="overridden Poly.as_expr"):
        groebner([polynomial], x)


def test_nonrational_poly_domain_is_refused_even_when_coefficients_look_exact():
    x = Symbol("real_poly_input_x")
    polynomial = Poly(x + 1, x, domain="RR")
    with pytest.raises(NotImplementedError, match="ZZ and QQ Poly"):
        groebner([polynomial], x)
