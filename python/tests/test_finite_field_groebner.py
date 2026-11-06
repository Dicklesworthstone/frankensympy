"""Complete prime-field ideals through the actual checked native extension.

Explicit-field basis and normal-form fixtures were independently evaluated
with isolated SymPy 1.14.0. Admission, retained-field inference and mutation
tests exercise this implementation's bounded source-preserving contract.
"""

import pytest

import sympy as sp
from sympy.polys.polyerrors import (
    CoercionFailed, DomainError, GeneratorsError, GeneratorsNeeded,
    UnificationFailed,
)


@pytest.mark.parametrize("polys", [False, True])
def test_prime_field_bases_have_the_exact_public_class_and_coherent_hashes(polys):
    from sympy.polys.polytools import GroebnerBasis

    x, y = sp.symbols("x y")
    basis = sp.groebner([x, y], x, y, modulus=5, polys=polys)
    equivalent = sp.groebner([2*x, 3*y], x, y, modulus=5, polys=polys)
    assert type(basis) is GroebnerBasis
    assert type(equivalent) is GroebnerBasis
    assert basis == equivalent and equivalent == basis
    original_hash = hash(basis)
    assert hash(equivalent) == original_hash
    assert {basis: "the same checked ideal"}[equivalent] == "the same checked ideal"
    assert len({basis, equivalent}) == 1
    assert basis.contains(x + y)
    assert hash(basis) == original_hash


def test_basis_equality_distinguishes_coefficient_fields_and_monomial_orders():
    x, y = sp.symbols("x y")
    finite = sp.groebner([x, y], x, y, modulus=5, order="lex")
    different_prime = sp.groebner([x, y], x, y, modulus=7, order="lex")
    rational = sp.groebner([x, y], x, y, domain="QQ", order="lex")
    different_order = sp.groebner([x, y], x, y, modulus=5, order="grlex")
    # The printed lists coincide; their typed polynomial-ring requests do not.
    for other in (different_prime, rational, different_order):
        assert finite.exprs == other.exprs
        assert finite != other and other != finite
    assert len({finite, different_prime, rational, different_order}) == 4


def test_field_basis_equality_retains_output_and_coefficient_lift_policies():
    x, y = sp.symbols("x y")
    expressions = sp.groebner([x, y], x, y, modulus=5, polys=False)
    polynomials = sp.groebner([x, y], x, y, modulus=5, polys=True)
    signed = sp.groebner([x, y], x, y, modulus=5, symmetric=True)
    unsigned = sp.groebner([x, y], x, y, modulus=5, symmetric=False)
    assert expressions.exprs == polynomials.exprs == signed.exprs == unsigned.exprs
    assert expressions != polynomials and polynomials != expressions
    assert signed != unsigned and unsigned != signed


def test_shallow_copy_preserves_the_public_basis_and_its_field_metadata():
    from copy import copy
    from sympy.polys.polytools import GroebnerBasis

    x, y = sp.symbols("x y")
    basis = sp.groebner([x + y, y**2 + 1], x, y, modulus=5,
                        order="grevlex", symmetric=False, polys=True)
    copied = copy(basis)
    # This is the shell's ordinary public-instance copy contract. It does not
    # claim that upstream's generic Basic pickle protocol preserves options.
    assert copied is not basis
    assert type(copied) is GroebnerBasis
    assert copied == basis and hash(copied) == hash(basis)
    assert str(copied.domain) == "GF(5)" and str(copied.order) == "grevlex"
    assert all(a is b for a, b in zip(copied.gens, basis.gens))
    assert all(type(polynomial) is sp.Poly and polynomial.get_modulus() == 5
               for polynomial in copied)
    assert copied.exprs == basis.exprs
    assert copied.contains(x + y)
    assert copied.reduce(x)[1] == sp.Poly(4*y, x, y, modulus=5, symmetric=False)
    assert copied.is_zero_dimensional


def test_retained_field_markers_prevent_domain_mutation_from_entering_rational_operations(monkeypatch):
    from sympy.core import _native
    from sympy.polys import polytools

    x, y = sp.symbols("x y")
    basis = sp.groebner([x, y], x, y, modulus=5)
    rational = sp.groebner([x, y], x, y, domain="QQ")
    basis.domain = "QQ"

    def forbidden(*args, **kwargs):
        raise AssertionError("inconsistent field metadata reached native mathematics")

    for name in (
        "groebner_reduce_native", "groebner_is_zero_dimensional_native",
        "groebner_fglm_native", "ideal_eliminate_native",
        "ideal_intersection_native", "ideal_quotient_native",
        "quotient_multiplication_native", "prime_field_groebner_native",
        "prime_field_groebner_zero_dimensional_native",
    ):
        monkeypatch.setattr(_native, name, forbidden)
    for operation in (
        lambda: basis.reduce(x),
        lambda: basis.contains(x),
        lambda: basis.is_zero_dimensional,
        lambda: basis.fglm("lex"),
        lambda: basis.eliminate(x),
        lambda: basis.multiplication_matrix(x),
        lambda: basis._native_basis(),
        lambda: basis._ideal_inputs(),
        lambda: polytools._IdealQuotientSnapshot(basis),
        lambda: rational.intersection(basis),
        lambda: rational.quotient(basis),
    ):
        with pytest.raises((DomainError, ValueError, NotImplementedError)):
            operation()


def test_characteristic_two_computes_a_proper_ideal_instead_of_the_rational_unit_ideal():
    x = sp.Symbol("field_groebner_counterexample")
    sources = [x**2 + 1, x + 1]
    rational = sp.groebner(sources, x, domain="QQ")
    finite = sp.groebner(sources, x, modulus=2)
    assert rational.exprs == [1]
    assert finite.exprs == [x + 1]
    assert str(finite.domain) == "GF(2)"
    assert finite.contains(x**2 + 1)
    assert not finite.contains(sp.S.One)
    assert finite.reduce(x**3) == ([x**2 + x + 1], sp.S.One)


@pytest.mark.parametrize("prime,expected_remainder", [(2, 1), (3, 1), (5, 2), (7, 0)])
@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_multivariate_bases_and_normal_forms_match_pinned_oracle(prime, expected_remainder, order):
    x, y = sp.symbols("x y")
    sources = [x*y - 1, y**2 - x]
    basis = sp.groebner(sources, x, y, modulus=prime, order=order)
    sign = 1 if prime == 2 else -1
    expected = ([x + sign*y**2, y**3 + sign] if order == "lex" else
                [x**2 + sign*y, x*y + sign, y**2 + sign*x])
    assert basis.exprs == expected
    assert list(basis) == expected
    assert basis.gens == (x, y) and basis.gens[0] is x and basis.gens[1] is y
    assert str(basis.order) == order and str(basis.domain) == "GF(%d)" % prime
    assert basis.is_zero_dimensional
    assert all(basis.contains(source) for source in sources)
    assert not basis.contains(x)
    dividend = x**3 + y**3 + 2*x*y + 3
    quotients, remainder = basis.reduce(dividend)
    assert remainder == expected_remainder
    assert len(quotients) == len(expected)
    reconstruction = sum(q*g for q, g in zip(quotients, expected)) + remainder
    assert sp.Poly(dividend - reconstruction, x, y, modulus=prime).is_zero
    assert basis.contains(dividend) is (expected_remainder == 0)
    assert basis.reduce(dividend, auto=False) == (quotients, remainder)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_three_variable_orders_match_independent_oracle(order):
    x, y, z = sp.symbols("x y z")
    expected = {
        "lex": [x - y*z, y**2 - z**2, y*z**2 - y, z**3 - z],
        "grlex": [z**3 - z, x**2 - z**2, x*y - z, x*z - y, y**2 - z**2, y*z - x],
        "grevlex": [z**3 - z, x**2 - z**2, x*y - z, y**2 - z**2, x*z - y, y*z - x],
    }
    basis = sp.groebner([x*y - z, x*z - y, y*z - x], x, y, z,
                        modulus=5, order=order)
    assert basis.exprs == expected[order]
    assert basis.is_zero_dimensional


def test_graded_orders_can_require_different_complete_polynomial_sets():
    x, y, z = sp.symbols("x y z")
    sources = [x**2 + y*z, x*y + z**2]
    grlex = sp.groebner(sources, x, y, z, modulus=5, order="grlex")
    grevlex = sp.groebner(sources, x, y, z, modulus=5, order="grevlex")
    assert grlex.exprs == [y**3*z + z**4, x*z**2 - y**2*z, x**2 + y*z, x*y + z**2]
    assert grevlex.exprs == [-x*z**2 + y**2*z, x**2 + y*z, x*y + z**2]
    assert len(grlex) == 4 and len(grevlex) == 3
    assert all(grlex.contains(p) and grevlex.contains(p) for p in sources)


@pytest.mark.parametrize("prime", [5, 65521])
@pytest.mark.parametrize("symmetric", [True, False])
def test_nonmonic_inputs_signed_lifts_and_largest_supported_prime(prime, symmetric):
    x, y = sp.symbols("x y")
    basis = sp.groebner([2*x + y, y**2 + 1], x, y,
                        modulus=sp.Integer(prime), symmetric=symmetric)
    half = (prime + 1)//2
    coefficient = half - prime if symmetric else half
    assert basis.exprs == [x + coefficient*y, y**2 + 1]
    assert basis.reduce(x) == ([sp.S.One, sp.S.Zero], ((prime - 1)//2)*y)
    assert [p.as_expr() for p in basis.polys] == basis.exprs
    assert basis.contains(2*x + y)


@pytest.mark.parametrize("case", ["empty", "zero", "unit", "positive", "finite", "redundant"])
def test_zero_unit_positive_dimension_and_redundant_input_ideals(case):
    x, y = sp.symbols("x y")
    sources, expected, dimension = {
        "empty": ([], [], False),
        "zero": ([5, 0], [], False),
        "unit": ([x, 2], [sp.S.One], False),
        "positive": ([x*y], [x*y], False),
        "finite": ([x + y, y**2 + 1], [x + y, y**2 + 1], True),
        "redundant": ([0, x + y, 2*x + 2*y, x + y], [x + y], False),
    }[case]
    basis = sp.groebner(sources, x, y, modulus=5)
    assert basis.exprs == expected
    assert basis.is_zero_dimensional is dimension
    assert basis.contains(sp.S.Zero)
    if case in ("empty", "zero"):
        assert basis.reduce(x + y) == ([], x + y)
        assert not basis.contains(x)
    if case == "unit":
        assert basis.reduce(x + 2*y) == ([x + 2*y], sp.S.Zero)
        assert basis.contains(x + y)


@pytest.mark.parametrize("source_kind", ["expressions", "poly", "purepoly", "mixed"])
@pytest.mark.parametrize("polys", [None, False, True])
def test_output_policy_and_field_poly_metadata_match_pinned_options(source_kind, polys):
    x, y = sp.symbols("x y")
    sources = {
        "expressions": [x + y],
        "poly": [sp.Poly(x + y, x, y, modulus=5)],
        "purepoly": [sp.PurePoly(x + y, x, y, modulus=5)],
        "mixed": [sp.Poly(x + y, x, y, modulus=5), x],
    }[source_kind]
    basis = sp.groebner(sources, x, y, modulus=5, polys=polys)
    expected_polys = source_kind != "expressions" if polys is None else polys
    assert all(type(p) is sp.Poly for p in basis.polys)
    assert all(p.gens == (x, y) and p.get_modulus() == 5 for p in basis.polys)
    assert [p.as_expr() for p in basis.polys] == basis.exprs
    assert all(isinstance(p, sp.Poly) is expected_polys for p in basis)
    quotients, remainder = basis.reduce(x + y + 1)
    assert all(isinstance(p, sp.Poly) is expected_polys for p in [*quotients, remainder])
    if expected_polys:
        assert remainder == sp.Poly(1, x, y, modulus=5)
    else:
        assert remainder == 1


@pytest.mark.parametrize("poly_type", [sp.Poly, sp.PurePoly])
def test_explicit_domain_and_retained_field_inputs_keep_the_requested_characteristic(poly_type):
    x, y = sp.symbols("x y")
    finite = poly_type(x*y - 1, x, y, modulus=2)
    expected = [x + y**2, y**3 + 1]
    assert sp.groebner([x*y - 1, y**2 - x], x, y, domain="GF(2)").exprs == expected
    # Retained field inference is this shell's explicit domain-preserving API.
    inferred = sp.groebner([finite, y**2 - x], x, y)
    assert inferred.exprs == expected
    assert str(inferred.domain) == "GF(2)"
    assert all(p.get_modulus() == 2 for p in inferred.polys)


def test_inferred_generators_preserve_typed_atoms_and_declared_zero_poly_ring():
    positive = sp.Symbol("same_field_generator", positive=True)
    negative = sp.Symbol("same_field_generator", negative=True)
    dummy = sp.Dummy("same_field_generator")
    sources = [positive - negative, negative - dummy]
    inferred = sp.groebner(sources, modulus=5)
    assert len(inferred.gens) == 3 and set(inferred.gens) == {positive, negative, dummy}
    assert all(any(g is original for original in (positive, negative, dummy)) for g in inferred.gens)
    assert all(inferred.contains(source) for source in sources)
    explicit = sp.groebner(sources, positive, negative, dummy, modulus=5)
    assert explicit.exprs == [positive - dummy, negative - dummy]
    assert not explicit.contains(positive)
    zero = sp.Poly(0, positive, negative, dummy, modulus=5)
    empty = sp.groebner([zero])
    assert set(empty.gens) == {positive, negative, dummy}
    assert empty.exprs == []


def test_exact_rational_cancellation_precedes_integer_to_field_mapping():
    x, y = sp.symbols("x y")
    repeated_halves = sp.Add(x/2, x/2, evaluate=False)
    canceled_denominator = sp.Mul(2, sp.Pow(2, -1, evaluate=False), x, evaluate=False)
    assert sp.groebner([repeated_halves, y], x, y, modulus=2).exprs == [x, y]
    assert sp.groebner([canceled_denominator, y], x, y, modulus=2).exprs == [x, y]
    for prime in (2, 5):
        with pytest.raises((CoercionFailed, ValueError)):
            sp.groebner([x/2, y], x, y, modulus=prime)
        basis = sp.groebner([x], x, y, modulus=prime)
        with pytest.raises((CoercionFailed, ValueError)):
            basis.reduce(y/2)


@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power", "cancellation", "undefined"])
def test_original_unsupported_children_are_checked_even_after_a_unit_generator(wrapper):
    x, y = sp.symbols("x y")
    invalid = sp.sin(y)
    if wrapper == "zero_product":
        hidden = sp.Mul(0, invalid, evaluate=False)
    elif wrapper == "zero_power":
        hidden = sp.Pow(invalid, 0, evaluate=False)
    elif wrapper == "cancellation":
        hidden = sp.Add(invalid, sp.Mul(-1, invalid, evaluate=False), evaluate=False)
    else:
        hidden = sp.Mul(0, sp.Pow(0, -1, evaluate=False), evaluate=False)
    for sources in ([hidden], [sp.S.One, hidden]):
        with pytest.raises((ValueError, NotImplementedError)):
            sp.groebner(sources, x, y, modulus=5)
    unit = sp.groebner([1], x, y, modulus=5)
    with pytest.raises((ValueError, NotImplementedError)):
        unit.contains(hidden)


def test_original_bounds_cannot_be_erased_by_modular_zero_or_unit_shortcuts():
    x, y = sp.symbols("x y")
    assert sp.groebner([5*x**64 + y], x, y, modulus=5).exprs == [y]
    for source in (5*x**33*y**32, sp.Integer(1 << 4097)*x,
                   sp.Pow(x, 10**9, evaluate=False)):
        with pytest.raises(ValueError):
            sp.groebner([1, source], x, y, modulus=5)
    with pytest.raises((ValueError, GeneratorsError)):
        sp.groebner([0], *sp.symbols("a b c d e f g h i"), modulus=5)
    with pytest.raises(ValueError):
        sp.groebner([0]*33, x, y, modulus=5)


@pytest.mark.parametrize("modulus", [0, 1, 4, 9, 65522, True, False, 5.0, "5"])
def test_invalid_fields_are_refused_even_for_the_zero_ideal(modulus):
    x = sp.Symbol("x")
    with pytest.raises((ValueError, CoercionFailed, DomainError)):
        sp.groebner([], x, modulus=modulus)


def test_mixed_fields_and_invalid_options_cannot_reach_the_rational_engine(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    five = sp.Poly(x, x, y, modulus=5)
    seven = sp.Poly(y, x, y, modulus=7)

    def forbidden(*args, **kwargs):
        raise AssertionError("a field request entered the rational basis engine")

    monkeypatch.setattr(_native, "groebner_basis_native", forbidden)
    with pytest.raises((DomainError, UnificationFailed, ValueError)):
        sp.groebner([five, seven], x, y)
    for options in ({"modulus": 5, "domain": "QQ"}, {"modulus": 5, "domain": "GF(7)"}):
        with pytest.raises((DomainError, NotImplementedError, ValueError)):
            sp.groebner([x, y], x, y, **options)
    for options in ({"method": "f5b"}, {"extension": True}, {"field": True}):
        with pytest.raises(NotImplementedError):
            sp.groebner([x, y], x, y, modulus=5, **options)
    with pytest.raises((TypeError, ValueError)):
        sp.groebner([x], x, modulus=5, polys=1)
    with pytest.raises(ValueError):
        sp.groebner([x], x, modulus=5, order="unknown")


def test_unsupported_or_inconsistent_generators_are_refused_before_shortcuts():
    x = sp.Symbol("x")
    nc = sp.Symbol("noncommutative_field_groebner", commutative=False)
    with pytest.raises((GeneratorsNeeded, GeneratorsError, ValueError)):
        sp.groebner([], modulus=5)
    with pytest.raises((GeneratorsError, ValueError)):
        sp.groebner([0], x, x, modulus=5)
    with pytest.raises((GeneratorsError, ValueError)):
        sp.groebner([1], x, nc, modulus=5)
    with pytest.raises((ValueError, GeneratorsError)):
        sp.groebner([sp.Mul(0, nc, evaluate=False)], x, modulus=5)


def test_public_basis_mutation_requires_a_complete_current_basis():
    x, y = sp.symbols("x y")
    basis = sp.groebner([x, y], x, y, modulus=5)
    basis.exprs[:] = [x*y - 1, y**2 - 1]
    # x-y belongs to this ideal, but these two divisors are not a complete
    # basis. An irreducible nonzero remainder would be a false nonmembership.
    for query in (lambda: basis.contains(x - y), lambda: basis.reduce(x - y),
                  lambda: basis.is_zero_dimensional):
        with pytest.raises(ValueError):
            query()
    basis.exprs[:] = [x - y, y**2 - 1]
    assert basis.contains(x - y)
    assert basis.is_zero_dimensional


def test_retained_poly_overrides_and_stale_source_storage_are_refused():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, modulus=5)
    calls = []

    def overridden():
        calls.append("as_expr")
        return sp.S.Zero

    polynomial.as_expr = overridden
    with pytest.raises(TypeError, match="as_expr"):
        sp.groebner([polynomial], x, y)
    assert calls == []
    held = sp.Add(x, y, evaluate=False)
    held._args = (x, x)
    with pytest.raises(ValueError):
        sp.groebner([held], x, y, modulus=5)


@pytest.mark.parametrize("mutation,source_shape", [
    ("missing_coordinate", "polynomial"),
    ("missing_coordinate", "held_zero"),
    ("missing_coordinate", "held_cancellation"),
    ("duplicate_coordinate", "polynomial"),
    ("duplicate_coordinate", "zero"),
])
def test_retained_poly_own_ring_is_checked_before_embedding_in_a_larger_ring(mutation, source_shape):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, modulus=5)
    basis = sp.groebner([x, y], x, y, modulus=5)
    if source_shape == "held_zero":
        polynomial._expr = sp.Mul(0, y, evaluate=False)
    elif source_shape == "held_cancellation":
        polynomial._expr = sp.Add(y, sp.Mul(-1, y, evaluate=False), evaluate=False)
    elif source_shape == "zero":
        polynomial._expr = sp.S.Zero
    polynomial._gens = (x,) if mutation == "missing_coordinate" else (x, x)
    # The requested ring contains y. That must not repair a retained Poly
    # whose own declaration excludes an original y child or duplicates x.
    for operation in (
        lambda: sp.groebner([polynomial], x, y, modulus=5),
        lambda: sp.groebner([sp.S.One, polynomial], x, y, modulus=5),
        lambda: basis.reduce(polynomial),
        lambda: basis.contains(polynomial),
    ):
        with pytest.raises((ValueError, GeneratorsError, UnificationFailed)):
            operation()


def test_effectful_poly_storage_is_refused_before_retained_field_dispatch(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, modulus=5)
    basis = sp.groebner([x, y], x, y, modulus=5)
    calls = []

    class HostileStorage(dict):
        def get(self, *args, **kwargs):
            calls.append("storage get")
            raise AssertionError("dispatch invoked effectful Poly storage")

        def __iter__(self):
            calls.append("storage iteration")
            raise AssertionError("dispatch iterated effectful Poly storage")

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed Poly storage reached native mathematics")

    polynomial.__dict__ = HostileStorage(polynomial.__dict__)
    monkeypatch.setattr(_native, "groebner_basis_native", forbidden)
    monkeypatch.setattr(_native, "prime_field_groebner_native", forbidden)
    for operation in (
        lambda: sp.groebner([polynomial], x, y),
        lambda: sp.groebner([polynomial], x, y, modulus=5),
        lambda: basis.reduce(polynomial),
        lambda: basis.contains(polynomial),
    ):
        with pytest.raises(TypeError):
            operation()
    assert calls == []


def test_unsupported_ideal_operations_do_not_reinterpret_a_prime_field_as_rationals():
    x, y = sp.symbols("x y")
    basis = sp.groebner([x + y, y**2 - 1], x, y, modulus=5)
    for operation in (lambda: basis.fglm("grlex"), lambda: basis.eliminate(x),
                      lambda: basis.invert(y), lambda: basis.multiplication_matrix(y),
                      lambda: basis.minpoly(y)):
        with pytest.raises(NotImplementedError):
            operation()


@pytest.mark.parametrize("target", ["input_container", "poly_metadata"])
def test_source_mutation_during_basis_generation_prevents_publication(monkeypatch, target):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, modulus=5)
    sources = [polynomial, y**2 - 1]
    original = _native.prime_field_groebner_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "input_container":
            sources[0] = x
        else:
            polynomial._modulus = 7
        return result

    monkeypatch.setattr(_native, "prime_field_groebner_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.groebner(sources, x, y, modulus=5)


@pytest.mark.parametrize("target", ["exprs", "gens", "domain", "order"])
def test_basis_metadata_and_divisors_remain_bound_through_reduction(monkeypatch, target):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    basis = sp.groebner([x + y, y**2 - 1], x, y, modulus=5)
    original = _native.prime_field_groebner_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "exprs":
            basis.exprs[0] = x
        elif target == "gens":
            basis.gens = (y, x)
        elif target == "domain":
            basis.domain = "GF(7)"
        else:
            basis.order = "grevlex"
        return result

    monkeypatch.setattr(_native, "prime_field_groebner_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.reduce(x**2 + y)


def test_basis_mutation_during_zero_dimension_check_is_not_published(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    basis = sp.groebner([x, y], x, y, modulus=5)
    original = _native.prime_field_groebner_zero_dimensional_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        basis.exprs[:] = [x*y]
        return result

    monkeypatch.setattr(_native, "prime_field_groebner_zero_dimensional_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _ = basis.is_zero_dimensional


def test_output_lifting_cannot_publish_after_source_mutation(monkeypatch):
    from sympy.polys import _finitefield_groebner

    x, y = sp.symbols("x y")
    sources = [x + y, y**2 - 1]
    original = _finitefield_groebner._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        sources.append(x)
        return result

    monkeypatch.setattr(_finitefield_groebner, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.groebner(sources, x, y, modulus=5)


@pytest.mark.parametrize("operation", ["basis", "reduce", "contains", "dimension"])
def test_native_refusal_is_terminal_without_restarting_over_another_domain(monkeypatch, operation):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    basis = sp.groebner([x, y], x, y, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise ValueError("independent prime-field verifier budget exhausted")

    def forbidden(*args, **kwargs):
        raise AssertionError("finite-field refusal retried in the rational engine")

    entrypoint = ("prime_field_groebner_zero_dimensional_native" if operation == "dimension"
                  else "prime_field_groebner_native")
    monkeypatch.setattr(_native, entrypoint, refuse)
    monkeypatch.setattr(_native, "groebner_basis_native", forbidden)
    monkeypatch.setattr(_native, "groebner_reduce_native", forbidden)
    with pytest.raises(ValueError, match="verifier budget"):
        if operation == "basis":
            sp.groebner([x, y], x, y, modulus=5)
        elif operation == "reduce":
            basis.reduce(x + 1)
        elif operation == "contains":
            basis.contains(x + 1)
        else:
            _ = basis.is_zero_dimensional
    assert calls == [1]


def test_effectful_containers_and_scalar_hooks_are_not_invoked():
    x = sp.Symbol("x")
    calls = []

    class HostileList(list):
        def __iter__(self):
            calls.append("iteration")
            raise AssertionError("source admission invoked an untrusted iterator")

    class HostileInteger:
        def __int__(self):
            calls.append("integer coercion")
            raise AssertionError("source admission invoked untrusted scalar coercion")

        def __index__(self):
            calls.append("index coercion")
            raise AssertionError("source admission invoked untrusted scalar coercion")

    with pytest.raises(TypeError):
        sp.groebner(HostileList([x]), x, modulus=5)
    with pytest.raises((CoercionFailed, TypeError, ValueError)):
        sp.groebner([x], x, modulus=HostileInteger())
    assert calls == []
