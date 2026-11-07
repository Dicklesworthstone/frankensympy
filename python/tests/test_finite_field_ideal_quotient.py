"""Finite prime-field ideal quotients through the actual public/native route.

Expected bases are hand-derived ideal identities. These extension methods are
not upstream SymPy APIs, and membership checks alone are not the comparator:
each positive fixture requires the complete stated reduced lexicographic basis.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr
from sympy.polys.polyerrors import CoercionFailed, DomainError, UnificationFailed
from sympy.polys.polytools import GroebnerBasis


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_finite_quotient_removes_one_factor_instead_of_saturating(prime):
    x, y = sp.symbols("modular_quotient_multiplicity_x modular_quotient_multiplicity_y")
    ideal = sp.groebner([x**3*y], x, y, modulus=prime)
    divisor = sp.groebner([x], y, x, modulus=prime, order="grevlex")
    result = ideal.quotient(divisor)
    assert type(result) is GroebnerBasis
    assert result.exprs == [x**2*y]
    assert result.gens == (x, y) and result.gens[0] is x and result.gens[1] is y
    assert result.domain == "GF(%d)" % prime and result.order == "lex"
    assert result.quotient(divisor).exprs == [x*y]
    assert ideal.saturate(x).exprs == [y]
    assert not result.contains(y) and not result.contains(x*y)
    assert ideal.exprs == [x**3*y] and divisor.exprs == [x]


@pytest.mark.parametrize("prime", [2, 5])
def test_nonmonomial_principal_quotient_recovers_every_generator(prime):
    x, y, z = sp.symbols("modular_quotient_linear_x modular_quotient_linear_y modular_quotient_linear_z")
    ideal = sp.groebner([(x+y)**2, (x+y)*z], x, y, z, modulus=prime, order="grevlex")
    divisor = sp.groebner([x+y], z, y, x, modulus=prime)
    result = ideal.quotient(divisor)
    assert result.gens == (x, y, z) and result.exprs == [x+y, z]
    assert all(ideal.contains(expression*(x+y)) for expression in result.exprs)
    assert not result.contains(1)


@pytest.mark.parametrize("prime", [2, 5])
def test_multiple_divisors_require_all_products_not_a_single_principal_quotient(prime):
    x, y = sp.symbols("modular_quotient_all_x modular_quotient_all_y")
    ideal = sp.groebner([x**2, y**2], x, y, modulus=prime)
    divisors = sp.groebner([y, x], y, x, modulus=prime, order="grlex")
    result = ideal.quotient(divisors)
    assert result.exprs == [x**2, x*y, y**2]
    assert result.contains(x*y) and not result.contains(x) and not result.contains(y)
    for expression in result.exprs:
        assert ideal.contains(expression*x) and ideal.contains(expression*y)
    assert sp.groebner([x*y], x, y, modulus=prime).quotient(divisors).exprs == [x*y]


def test_characteristic_two_changes_the_finite_quotient_and_zero_divisors():
    x = sp.Symbol("modular_quotient_characteristic_x")
    ideal = sp.groebner([x**2+1], x, modulus=2)
    divisor = sp.groebner([x+1], x, modulus=2)
    assert ideal.quotient(divisor).exprs == [x+1]
    assert ideal.quotient(divisor).quotient(divisor).exprs == [1]
    assert ideal.saturate(x+1).exprs == [1]
    rational = sp.groebner([x**2+1], x, domain="QQ")
    assert rational.quotient(sp.groebner([x+1], x, domain="QQ")).exprs == [x**2+1]
    # The current divisor source is nonzero over QQ but zero in the retained F_2.
    divisor.exprs[:] = [2*x]
    assert ideal.quotient(divisor).exprs == [1]


def test_quotient_is_an_algebraic_ideal_without_finite_point_equations():
    x = sp.Symbol("modular_quotient_algebraic_x")
    ideal = sp.groebner([x**2+1], x, modulus=3)
    # No F_3 point solves x**2+1=0, but this remains a proper polynomial ideal.
    result = ideal.quotient(sp.groebner([x], x, modulus=3))
    assert result.exprs == [x**2+1]
    assert not result.contains(1) and not result.contains(x**3-x)
    assert result.eliminate(x).exprs == []


def test_zero_and_unit_divisor_laws_are_complete():
    x, y = sp.symbols("modular_quotient_edge_x modular_quotient_edge_y")
    zero = sp.groebner([], x, y, modulus=5)
    unit = sp.groebner([1], x, y, modulus=5)
    ideal = sp.groebner([x*y-1], x, y, modulus=5)
    assert ideal.quotient(zero).exprs == zero.quotient(zero).exprs == [1]
    assert zero.quotient(ideal).exprs == []
    assert ideal.quotient(unit).exprs == ideal.exprs
    assert unit.quotient(ideal).exprs == [1]
    assert ideal.quotient(ideal).exprs == [1]
    zero.exprs[:] = [0, 5, 10*x]
    assert ideal.quotient(zero).exprs == [1]


@pytest.mark.parametrize("polys", [False, True])
def test_constant_field_quotients_preserve_the_left_return_policy(polys):
    x = sp.Symbol("modular_quotient_constant_x")
    zero = sp.groebner([x**2+1], x, modulus=3, polys=polys).eliminate(x)
    unit = sp.groebner([1], x, modulus=3, polys=polys).eliminate(x)
    for left, right, expected in ((zero, zero, [1]), (zero, unit, []),
                                  (unit, zero, [1]), (unit, unit, [1])):
        result = left.quotient(right)
        assert type(result) is GroebnerBasis
        assert result.gens == () and result.exprs == expected
        assert result.domain == "GF(3)" and result.order == "lex"
        if expected:
            assert result == unit and hash(result) == hash(unit)
            if polys:
                assert type(result[0]) is sp.Poly and result[0].gens == ()
                assert result[0].as_expr() == 1 and result[0].get_modulus() == 3
                assert result[0] == 4
                quotients, remainder = result.reduce(7)
                assert quotients[0].as_expr() == 1 and remainder.is_zero
            else:
                assert list(result) == [1] and result.reduce(7) == ([1], 0)


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quotient_keeps_left_field_order_and_coefficient_lift_policies(polys, symmetric):
    x, y, z = sp.symbols("modular_quotient_policy_x modular_quotient_policy_y modular_quotient_policy_z")
    ideal = sp.groebner([(x-y)**2, (x-y)*z], x, y, z, modulus=5,
                       order="grevlex", polys=polys, symmetric=symmetric)
    divisors = sp.groebner([x-y], z, y, x, modulus=5, order="grlex",
                          polys=not polys, symmetric=not symmetric)
    result = ideal.quotient(divisors)
    assert result.exprs == [x-y if symmetric else x+4*y, z]
    assert result.order == "lex" and result.domain == "GF(5)"
    assert all(a is b for a, b in zip(result.gens, ideal.gens))
    expected = sp.groebner([x-y, z], x, y, z, modulus=5, polys=polys, symmetric=symmetric)
    assert result == expected and hash(result) == hash(expected)
    if polys:
        assert all(type(value) is sp.Poly and value.get_modulus() == 5 for value in result)
        assert result.reduce(x)[1] == sp.Poly(y, x, y, z, modulus=5, symmetric=symmetric)
    else:
        assert list(result) == result.exprs and result.reduce(x)[1] == y


def test_current_sources_keep_zero_duplicate_and_nonreduced_divisor_constraints():
    x, y = sp.symbols("modular_quotient_current_x modular_quotient_current_y")
    ideal = sp.groebner([x], x, y, modulus=5)
    ideal.exprs[:] = [2*x*y, 0, 4*x*y]
    divisors = sp.groebner([1], y, x, modulus=5)
    divisors.exprs[:] = [0, 3*x, 5, x]
    assert ideal.quotient(divisors).exprs == [y]
    divisors.exprs[:] = [0, 3*x, x*y, x]
    assert ideal.quotient(divisors).exprs == [y]


def test_typed_same_printed_and_auxiliary_named_axes_are_preserved():
    name = "_fsym_prime_field_ideal_parameter_0"
    plain = sp.Symbol(name)
    assumed = sp.Symbol(name, real=True)
    dummy = sp.Dummy(name)
    generators = (dummy, assumed, plain)
    ideal = sp.groebner([plain**2*assumed*dummy], *generators, modulus=5)
    divisor = sp.groebner([plain], *reversed(generators), modulus=5)
    result = ideal.quotient(divisor)
    assert result.exprs == [plain*assumed*dummy]
    assert all(a is b for a, b in zip(result.gens, generators))
    assert all(any(symbol is owner for symbol in result.exprs[0].free_symbols)
               for owner in generators)


def test_field_and_typed_ring_mismatches_refuse_in_both_directions(monkeypatch):
    x = sp.Symbol("modular_quotient_mismatch")
    foreign = sp.Symbol("modular_quotient_mismatch", real=True)
    ideal = sp.groebner([x], x, modulus=5)
    others = (sp.groebner([x], x, modulus=7),
              sp.groebner([x], x, domain="QQ"),
              sp.groebner([foreign], foreign, modulus=5))

    def forbidden(*args, **kwargs):
        raise AssertionError("mixed domain or typed ring reached ideal arithmetic")

    monkeypatch.setattr(_native, "prime_field_ideal_quotient_native", forbidden)
    monkeypatch.setattr(_native, "ideal_quotient_native", forbidden)
    for other in others:
        with pytest.raises(UnificationFailed):
            ideal.quotient(other)
        with pytest.raises(UnificationFailed):
            other.quotient(ideal)


@pytest.mark.parametrize("attribute,value", [("domain", "QQ"), ("_modulus", 7),
                                             ("_symmetric", 1), ("_polys", 1)])
def test_retained_field_metadata_cannot_route_a_quotient_into_qq(monkeypatch, attribute, value):
    x = sp.Symbol("modular_quotient_metadata_" + attribute)
    ideal = sp.groebner([x], x, modulus=5)
    finite = sp.groebner([1], x, modulus=5)
    rational = sp.groebner([1], x, domain="QQ")
    setattr(ideal, attribute, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid field metadata reached ideal arithmetic")

    monkeypatch.setattr(_native, "prime_field_ideal_quotient_native", forbidden)
    monkeypatch.setattr(_native, "ideal_quotient_native", forbidden)
    for other in (finite, rational):
        with pytest.raises(DomainError):
            ideal.quotient(other)
        with pytest.raises(DomainError):
            other.quotient(ideal)


@pytest.mark.parametrize("kind", ["reciprocal", "rational", "float", "unknown"])
@pytest.mark.parametrize("shape", ["zero_product", "zero_power"])
def test_every_original_source_is_admitted_before_zero_or_unit_shortcuts(kind, shape):
    x = sp.Symbol("modular_quotient_hidden_" + kind + "_" + shape)
    bad = {"reciprocal": sp.Pow(x, -1, evaluate=False),
           "rational": sp.Rational(1, 2), "float": sp.Float("1.0"),
           "unknown": sp.sin(x)}[kind]
    hidden = (sp.Mul(0, bad, evaluate=False) if shape == "zero_product"
              else sp.Pow(bad, 0, evaluate=False))
    first = sp.groebner([1], x, modulus=5)
    first.exprs[:] = [1, hidden]
    zero = sp.groebner([], x, modulus=5)
    unit = sp.groebner([1], x, modulus=5)
    for left, right in ((first, zero), (unit, first), (zero, first)):
        with pytest.raises((ValueError, CoercionFailed)):
            left.quotient(right)


@pytest.mark.parametrize("side", ["numerator", "divisor"])
def test_each_retained_poly_must_belong_to_its_own_original_ring(side):
    x, y = sp.symbols("modular_quotient_poly_x modular_quotient_poly_y")
    original = sp.Poly(0, x, modulus=5)
    original._expr = sp.Add(y, -y, evaluate=False)
    first = sp.groebner([x], x, y, modulus=5)
    second = sp.groebner([], y, x, modulus=5)
    if side == "numerator":
        first.exprs[:] = [original]
    else:
        second.exprs[:] = [original]
    with pytest.raises(ValueError):
        first.quotient(second)
    foreign = sp.Poly(0, x, y, modulus=7)
    second.exprs[:] = [foreign]
    with pytest.raises(UnificationFailed):
        first.quotient(second)


def test_unused_poly_axes_cannot_be_rebound_by_mutating_assumptions(monkeypatch):
    x = sp.Symbol("modular_quotient_unused")
    foreign = sp.Symbol("modular_quotient_unused", real=True)
    ideal = sp.groebner([x], x, modulus=5)
    divisor = sp.groebner([], x, modulus=5)
    divisor.exprs[:] = [sp.Poly(0, foreign, modulus=5)]
    monkeypatch.setattr(foreign, "_assumptions", dict(x._assumptions))
    with pytest.raises(ValueError, match="identity"):
        ideal.quotient(divisor)


@pytest.mark.parametrize("change", ["left_list", "right_list", "left_gens", "right_gens",
                                    "domain", "modulus", "symmetric", "polys",
                                    "held_children", "native_value", "assumptions"])
def test_original_storage_remains_bound_through_native_quotient_verification(monkeypatch, change):
    x, y = sp.symbols("modular_quotient_mutation_" + change + "_x modular_quotient_mutation_" + change + "_y")
    held = sp.Add(x, y, evaluate=False)
    first = sp.groebner([x], x, y, modulus=5)
    first.exprs[:] = [held]
    second = sp.groebner([1], y, x, modulus=5)
    original = _native.prime_field_ideal_quotient_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "left_list":
            first.exprs.append(x)
        elif change == "right_list":
            second.exprs.append(x)
        elif change == "left_gens":
            first.gens = (y, x)
        elif change == "right_gens":
            second.gens = (x, y)
        elif change == "domain":
            second.domain = "GF(7)"
        elif change == "modulus":
            first._modulus = 7
        elif change == "symmetric":
            second._symmetric = False
        elif change == "polys":
            first._polys = True
        elif change == "held_children":
            held._args = (x, x)
        elif change == "native_value":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_quotient_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.quotient(second)


def test_poly_metadata_is_retained_through_native_quotient_verification(monkeypatch):
    x = sp.Symbol("modular_quotient_poly_mutation")
    ideal = sp.groebner([x**2], x, modulus=5)
    divisor = sp.groebner([x], x, modulus=5)
    polynomial = sp.Poly(x, x, modulus=5)
    divisor.exprs[:] = [polynomial]
    original = _native.prime_field_ideal_quotient_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        polynomial._modulus = 7
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_quotient_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        ideal.quotient(divisor)


def test_both_operands_are_captured_before_the_first_preflight(monkeypatch):
    x = sp.Symbol("modular_quotient_preflight")
    first = sp.groebner([x], x, modulus=5)
    second = sp.groebner([], x, modulus=5)
    original = _native.ideal_source_preflight_native
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            calls.append(1)
            second.exprs.append(1)
        return result

    monkeypatch.setattr(_native, "ideal_source_preflight_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.quotient(second)
    assert calls == [1]


def test_lifting_a_quotient_cannot_publish_after_divisor_mutation(monkeypatch):
    from sympy.polys import _finitefield_ideals

    x = sp.Symbol("modular_quotient_lift")
    first = sp.groebner([x**2], x, modulus=5)
    second = sp.groebner([x], x, modulus=5)
    original = _finitefield_ideals._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        second.exprs[:] = [1]
        return result

    monkeypatch.setattr(_finitefield_ideals, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.quotient(second)


@pytest.mark.parametrize("failure", [ValueError, MemoryError])
def test_operational_failure_is_terminal_without_qq_or_fresh_modular_retry(monkeypatch, failure):
    x = sp.Symbol("modular_quotient_terminal")
    first = sp.groebner([x**2], x, modulus=5)
    second = sp.groebner([x], x, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure("independent quotient verifier budget exhausted or cancelled")

    def forbidden(*args, **kwargs):
        raise AssertionError("quotient refusal restarted in another arithmetic call")

    monkeypatch.setattr(_native, "prime_field_ideal_quotient_native", refuse)
    for name in ("ideal_quotient_native", "prime_field_ideal_intersection_native",
                 "prime_field_groebner_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(failure, match="budget exhausted or cancelled"):
        first.quotient(second)
    assert calls == [1]


def test_closed_operand_source_and_container_admission_never_calls_user_hooks():
    x = sp.Symbol("modular_quotient_no_hooks")
    ideal = sp.groebner([x], x, modulus=5)
    calls = []

    class HostileMeta(type):
        def __eq__(self, other):
            calls.append("metaclass equality")
            raise AssertionError("untrusted metaclass equality")

        __hash__ = type.__hash__

    class HostileBasis(GroebnerBasis, metaclass=HostileMeta):
        def __getattribute__(self, name):
            calls.append("basis attributes")
            raise AssertionError("untrusted basis attributes")

    class HostileList(list):
        def __iter__(self):
            calls.append("source iteration")
            raise AssertionError("untrusted source iteration")

    hostile = object.__new__(HostileBasis)
    with pytest.raises(TypeError):
        ideal.quotient(hostile)
    with pytest.raises(TypeError):
        GroebnerBasis.quotient(hostile, ideal)
    other = sp.groebner([], x, modulus=5)
    other.exprs = HostileList([x])
    with pytest.raises(ValueError):
        ideal.quotient(other)
    polynomial = sp.Poly(x, x, modulus=5)

    def as_expr():
        calls.append("Poly.as_expr")
        raise AssertionError("untrusted Poly protocol")

    polynomial.as_expr = as_expr
    other.exprs = [polynomial]
    with pytest.raises(TypeError, match="as_expr"):
        ideal.quotient(other)
    assert calls == []


def test_quotient_bounds_distinguish_zero_divisors_after_source_admission():
    axes = sp.symbols("modular_quotient_bound_x0:8")
    x = axes[0]
    wide = sp.groebner([x], *axes, modulus=5)
    zero = sp.groebner([], *reversed(axes), modulus=5)
    wide.exprs[:] = [x] * 32
    zero.exprs[:] = [5*x] * 32
    # The combined64 sources are permitted; every divisor becomes zero in F_5.
    result = wide.quotient(zero)
    assert result.exprs == [1] and result.gens == axes
    zero.exprs[:] = [x]
    with pytest.raises(ValueError):
        wide.quotient(zero)
    wide.exprs[:] = [x]
    with pytest.raises(ValueError):
        wide.quotient(zero)
    bounded = sp.groebner([x**2], *axes[:-1], modulus=5)
    divisor = sp.groebner([x], *axes[:-1], modulus=5)
    bounded.exprs[:] = [x**2] * 31
    assert bounded.quotient(divisor).exprs == [x]
    bounded.exprs.append(x**2)
    with pytest.raises(ValueError):
        bounded.quotient(divisor)
    bounded.exprs.append(x**2)
    with pytest.raises(ValueError):
        bounded.quotient(divisor)
    bounded.exprs[:] = [x**2]
    divisor.exprs[:] = [0] * 33
    with pytest.raises(ValueError):
        bounded.quotient(divisor)
