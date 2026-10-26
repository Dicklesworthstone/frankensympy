"""Checked public Groebner operations; require the actual native extension."""

import pytest

from sympy import Poly, Pow, Rational, Symbol, expand, groebner


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
