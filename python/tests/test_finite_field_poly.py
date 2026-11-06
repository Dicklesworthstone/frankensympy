"""Prime-field Poly semantics through the actual checked native bridge.

Coefficient fixtures below were obtained from the pinned SymPy 1.14.0 oracle.
These are native integration tests, not a Python arithmetic stand-in.
"""

import pytest

import sympy as sp
from sympy.core import _native
from sympy.polys import _finitefield
from sympy.polys.polyerrors import (
    CoercionFailed, DomainError, ExactQuotientFailed, GeneratorsError,
    GeneratorsNeeded, NotInvertible, UnificationFailed,
)


CASES = [
    dict(p=2, a=[1, 0, 1], b=[1, 1], add=[1, 1, 0], sub=[1, 1, 0],
         mul=[1, 1, 1, 1], gcd=[1, 1], lcm=[1, 0, 1], compose=[1, 0, 0],
         divrem=[[1, 1], [0]], gcdex=[[0], [1], [1, 1]], diff=[0], shift=[1, 0, 1]),
    dict(p=3, a=[2, 0, 1, 2], b=[1, 2, 1], add=[-1, 1, 0, 0], sub=[-1, -1, -1, 1],
         mul=[-1, 1, 0, 1, -1, -1], gcd=[1], lcm=[1, -1, 0, -1, 1, 1],
         compose=[-1, 0, 0, 1, 1, -1, -1], divrem=[[-1, -1], [1, 0]],
         gcdex=[[-1, 1], [-1, 0, -1], [1]], diff=[1], shift=[-1, 0, 1, -1]),
    dict(p=5, a=[2, 4, 0, 3], b=[3, 0, 1], add=[2, 2, 0, -1], sub=[2, 1, 0, 2],
         mul=[1, 2, 2, -2, 0, -2], gcd=[1], lcm=[1, 2, 2, -2, 0, -2],
         compose=[-1, 0, 0, 0, 2, 0, -1], divrem=[[-1, -2], [1, 0]],
         gcdex=[[2, 0], [2, -1, 1], [1]], diff=[1, -2, 0], shift=[2, 1, 0, 0]),
    dict(p=7, a=[1, 0, 0, 0, 2], b=[2, 1, 3], add=[1, 0, 2, 1, -2],
         sub=[1, 0, -2, -1, -1], mul=[2, 1, 3, 0, -3, 2, -1], gcd=[1],
         lcm=[1, -3, -2, 0, 2, 1, 3], compose=[2, -3, 1, -2, 2, -3, -3, 3, -1],
         divrem=[[-3, -2, 2], [-3, 3]], gcdex=[[-3, -1], [-2, -2, -3, 1], [1]],
         diff=[-3, 0, 0, 0], shift=[1, 1, 3, -3, -3]),
]


@pytest.mark.parametrize("case", CASES, ids=["GF2", "GF3", "GF5", "GF7"])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "gcd", "lcm", "compose"])
def test_arithmetic_matches_pinned_coefficients(case, operation):
    x = sp.Symbol("x")
    left = sp.Poly.from_list(case["a"], x, modulus=case["p"])
    right = sp.Poly.from_list(case["b"], x, modulus=case["p"])
    result = getattr(left, operation)(right)
    assert result.all_coeffs() == case[operation]
    assert result.get_modulus() == case["p"]
    assert str(result.domain) == "GF(%d)" % case["p"]
    assert result.gen is x


@pytest.mark.parametrize("case", CASES, ids=["GF2", "GF3", "GF5", "GF7"])
def test_division_bezout_derivative_and_shift_match_pinned_coefficients(case):
    x = sp.Symbol("x")
    left = sp.Poly.from_list(case["a"], x, modulus=case["p"])
    right = sp.Poly.from_list(case["b"], x, modulus=case["p"])
    quotient, remainder = left.div(right)
    assert [q.all_coeffs() for q in (quotient, remainder)] == case["divrem"]
    assert quotient * right + remainder == left
    s, t, h = left.gcdex(right)
    assert [q.all_coeffs() for q in (s, t, h)] == case["gcdex"]
    assert s * left + t * right == h
    assert left.diff().all_coeffs() == case["diff"]
    assert left.shift(2).all_coeffs() == case["shift"]


def test_constructor_field_alias_inference_and_integer_coefficient_normalization():
    x = sp.Symbol("field_constructor_x", positive=True)
    explicit = sp.Poly(7*x**2 + 4*x + 8, x, modulus=5)
    inferred = sp.Poly(7*x**2 + 4*x + 8, domain="GF(5)")
    assert explicit == inferred
    assert inferred.gen is x
    assert explicit.all_coeffs() == [2, -1, -2]
    assert explicit.as_dict() == {(2,): 2, (1,): -1, (0,): -2}
    assert repr(explicit).endswith("modulus=5)")
    assert sp.Poly(explicit).get_modulus() == 5
    assert sp.Poly.from_poly(explicit).get_modulus() == 5
    assert sp.Poly(explicit, modulus=3).all_coeffs() == [-1, -1, 1]
    half_sum = sp.Add(sp.Rational(1, 2), sp.Rational(1, 2), evaluate=False)
    assert sp.Poly(sp.Add(x, half_sum, evaluate=False), x, modulus=5).all_coeffs() == [1, 1]


def test_signed_and_unsigned_lifts_preserve_the_same_field_identity():
    x = sp.Symbol("x")
    signed = sp.Poly(4*x + 3, x, modulus=5)
    unsigned = sp.Poly(4*x + 3, x, modulus=5, symmetric=False)
    assert signed.all_coeffs() == [-1, -2]
    assert unsigned.all_coeffs() == [4, 3]
    assert signed == unsigned and hash(signed) == hash(unsigned)
    assert (-unsigned).all_coeffs() == [1, 2]
    assert (unsigned + 4).all_coeffs() == [4, 2]
    assert (2 - unsigned).all_coeffs() == [1, 4]
    assert unsigned.monic().all_coeffs() == [1, 2]
    assert unsigned.eval(2) == 1


def test_characteristic_changes_degree_derivative_and_gcd():
    x = sp.Symbol("x")
    polynomial = sp.Poly(x**2 + 1, x, domain="GF(2)")
    divisor = sp.Poly(x + 1, x, modulus=2)
    assert polynomial.diff().is_zero
    assert polynomial.gcd(divisor) == divisor
    assert sp.Poly(2*x**7 + 4*x, x, modulus=2).degree() == -sp.oo
    assert sp.Poly(0, x, modulus=2).all_coeffs() == [0]
    assert sp.Poly(0, x, modulus=2).coeffs() == [0]
    assert sp.Poly(0, x, modulus=2).monic().is_zero


def test_derivative_orders_use_poly_generator_specifications_and_one_native_call(monkeypatch):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x**4 + x**2 + 1, x, modulus=5)
    original = _native.prime_field_polynomial_native
    calls = []

    def checked(*args, **kwargs):
        calls.append((args[4], args[5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_polynomial_native", checked)
    result = polynomial.diff((x, 2))
    assert calls == [("diff", 2)]
    assert result.all_coeffs() == [2, 0, 2]
    assert polynomial.diff(0) == polynomial.diff(x)
    assert polynomial.diff(x, x) == result
    assert polynomial.diff((x, 0)) == polynomial
    with pytest.raises(GeneratorsError):
        polynomial.diff(2)
    with pytest.raises(GeneratorsError):
        polynomial.diff(x, 2)


def test_quotient_inverse_nonunits_and_zero_quotient_ring():
    x = sp.Symbol("x")
    polynomial = sp.Poly(x, x, modulus=5)
    modulus = sp.Poly(x**2 + 1, x, modulus=5)
    inverse = polynomial.invert(modulus)
    assert inverse.all_coeffs() == [-1, 0]
    assert (polynomial * inverse).rem(modulus).is_one
    assert sp.invert(polynomial, modulus) == inverse
    assert sp.Poly(0, x, modulus=5).invert(sp.Poly(1, x, modulus=5)).is_zero
    with pytest.raises(NotInvertible):
        sp.Poly(x + 1, x, modulus=2).invert(sp.Poly(x**2 + 1, x, modulus=2))
    with pytest.raises(ZeroDivisionError):
        polynomial.div(sp.Poly(0, x, modulus=5))
    with pytest.raises(ExactQuotientFailed):
        modulus.exquo(polynomial)
    assert (polynomial * modulus).exquo(polynomial) == modulus


def test_inseparable_factors_square_free_groups_and_monic_radical():
    x = sp.Symbol("x")
    polynomial = sp.Poly(2*(x + 1)**6*(x**2 + 1)**3, x, modulus=3)
    scale, factors = polynomial.factor_list()
    assert scale == -1
    assert [(factor.all_coeffs(), count) for factor, count in factors] == [([1, 1], 6), ([1, 0, 1], 3)]
    scale, square_free = polynomial.sqf_list()
    assert scale == -1
    assert [(factor.all_coeffs(), count) for factor, count in square_free] == [([1, 0, 1], 3), ([1, 1], 6)]
    assert polynomial.sqf_part().all_coeffs() == [1, 1, 1, 1]
    assert sp.Poly(0, x, modulus=3).sqf_part().is_one
    assert sp.Poly(2, x, modulus=3).sqf_part().is_one
    assert all(factor.get_modulus() == 3 for factor, _ in factors + square_free)
    assert sp.factor_list(polynomial) == polynomial.factor_list()
    assert sp.sqf_list(polynomial) == polynomial.sqf_list()
    assert sp.sqf_part(polynomial) == polynomial.sqf_part()


def test_frobenius_square_free_mode_uses_a_single_checked_bridge(monkeypatch):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x**6 + 1, x, modulus=2)
    original = _native.prime_field_polynomial_factor_native
    modes = []

    def checked(*args, **kwargs):
        modes.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_polynomial_factor_native", checked)
    scale, factors = polynomial.sqf_list()
    assert modes == ["sqf"]
    assert scale == 1 and [(factor.all_coeffs(), count) for factor, count in factors] == [([1, 0, 0, 1], 2)]
    assert polynomial.sqf_part().all_coeffs() == [1, 0, 0, 1]
    assert modes == ["sqf", "sqf_part"]


def test_scalar_queries_primitive_content_coefficients_and_power():
    x = sp.Symbol("x")
    polynomial = sp.Poly(2*x**3 + 4*x, x, modulus=5)
    assert polynomial.LC() == 2 and polynomial.TC() == 0 and polynomial.EC() == -1
    assert sp.EC(polynomial) == -1
    assert polynomial.nth(3) == 2 and polynomial.nth(2) == 0 and polynomial.nth(100) == 0
    with pytest.raises(IndexError):
        polynomial.nth(-1)
    assert polynomial.content() == 1
    assert polynomial.primitive() == (1, polynomial)
    zero = sp.Poly(0, x, modulus=5)
    assert zero.primitive() == (0, zero)
    assert polynomial**0 == sp.Poly(1, x, modulus=5)
    assert sp.Poly(x + 1, x, modulus=2)**4 == sp.Poly(x**4 + 1, x, modulus=2)
    assert sp.Poly(x + 3, x, modulus=5).eval(4) == 2
    with pytest.raises(UnificationFailed):
        polynomial.eval(sp.Rational(1, 2))
    with pytest.raises(UnificationFailed):
        polynomial.eval(polynomial)


def test_module_entry_points_retain_explicit_field_and_poly_output():
    x = sp.Symbol("x")
    assert sp.gcd(x**2 + 1, x + 1, x, modulus=2) == x + 1
    assert sp.lcm(x**2 + 1, x + 1, x, domain="GF(2)") == x**2 + 1
    assert sp.div(x**2 + 1, x + 1, x, modulus=2) == (x + 1, 0)
    assert sp.quo(x**2 + 1, x + 1, x, modulus=2) == x + 1
    assert sp.rem(x**2 + 1, x + 1, x, modulus=2) == 0
    assert sp.invert(x, x**2 + 1, x, modulus=5) == -x
    assert sp.gcdex(x, x**2 + 1, x, modulus=5) == (-x, 1, 1)
    assert sp.compose(x**2 + 1, 2*x + 1, x, modulus=3) == x**2 + x - 1
    quotient, remainder = sp.div(x**2 + 1, x + 1, x, modulus=2, polys=True)
    assert quotient.get_modulus() == remainder.get_modulus() == 2
    left, right = sp.Poly(x**2 + 1, x, modulus=2), sp.Poly(x + 1, x, modulus=2)
    assert sp.gcd(left, right).get_modulus() == 2
    assert sp.div(left, right) == (right, sp.Poly(0, x, modulus=2))


@pytest.mark.parametrize("other", ["different_prime", "QQ", "ZZ", "expression", "rational", "float", "bool"])
def test_mixed_domains_and_implicit_symbolic_coercions_are_refused(other):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x + 1, x, modulus=5)
    values = {
        "different_prime": sp.Poly(x + 1, x, modulus=3),
        "QQ": sp.Poly(x + 1, x, domain="QQ"),
        "ZZ": sp.Poly(x + 1, x, domain="ZZ"),
        "expression": x + 1, "rational": sp.Rational(1, 2), "float": 1.0, "bool": True,
    }
    with pytest.raises(UnificationFailed):
        polynomial + values[other]
    with pytest.raises(UnificationFailed):
        polynomial.div(values[other])


def test_equality_hashing_and_pure_polynomial_generator_independence():
    x, y = sp.symbols("x y")
    first = sp.Poly(x + 1, x, modulus=5)
    assert first != sp.Poly(x + 1, x, modulus=3)
    assert first != sp.Poly(x + 1, x, domain="ZZ")
    assert first != sp.Poly(y + 1, y, modulus=5)
    pure = sp.PurePoly(y + 1, y, modulus=5)
    assert first == pure and pure == first
    assert hash(first) == hash(pure)
    assert first == x + 1
    assert first != True and first != 1.5 and first != sp.sin(x) and first != y + 1


@pytest.mark.parametrize("operation", ["integrate", "roots", "all_roots", "real_roots", "nroots", "sturm", "decompose"])
def test_unsupported_field_operations_refuse_instead_of_using_rational_algorithms(operation):
    x = sp.Symbol("x")
    with pytest.raises(NotImplementedError, match="prime-field"):
        getattr(sp.Poly(x**2 + 1, x, modulus=2), operation)()


def test_no_rational_arithmetic_fallback_is_called(monkeypatch):
    x = sp.Symbol("x")
    left, right = sp.Poly(x**2 + 1, x, modulus=2), sp.Poly(x + 1, x, modulus=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("rational polynomial arithmetic was invoked")

    for name in ("poly_div_rem_native", "poly_gcd_native", "poly_gcdex_native", "poly_lcm_native",
                 "poly_monic_native", "poly_compose_native", "poly_factor_list_native", "poly_sqf_list_native"):
        monkeypatch.setattr(_native, name, forbidden)
    assert left.gcd(right) == right
    assert left.div(right) == (right, sp.Poly(0, x, modulus=2))
    assert left.diff().is_zero
    assert left.factor_list()[1] == [(right, 2)]


@pytest.mark.parametrize("modulus", [0, 1, 4, 9, 65522, 65537, -5])
def test_nonprime_and_out_of_range_moduli_are_refused(modulus):
    x = sp.Symbol("x")
    with pytest.raises(ValueError):
        sp.Poly(x + 1, x, modulus=modulus)


def test_constructor_coercion_and_regime_limits_are_explicit():
    x, y = sp.symbols("x y")
    for expression in (x + sp.Rational(1, 2), x + sp.Rational(1, 5)):
        with pytest.raises(CoercionFailed):
            sp.Poly(expression, x, modulus=5)
    with pytest.raises(GeneratorsNeeded):
        sp.Poly(3, modulus=5)
    with pytest.raises(GeneratorsNeeded):
        sp.Poly.from_list([1, 2], modulus=5)
    with pytest.raises(GeneratorsError):
        sp.Poly(x + y, x, y, modulus=5)
    with pytest.raises((ValueError, NotImplementedError)):
        sp.Poly(x + y, modulus=5)
    with pytest.raises(ValueError):
        sp.Poly(x**65, x, modulus=5)
    with pytest.raises(DomainError):
        sp.Poly(x + 1, x, modulus=5, domain="GF(7)")
    with pytest.raises(TypeError):
        sp.Poly(x + 1, x, modulus=5, symmetric=1)


def test_typed_generators_survive_results_and_cannot_be_conflated():
    positive = sp.Symbol("field_typed_x", positive=True)
    negative = sp.Symbol("field_typed_x", negative=True)
    dummy = sp.Dummy("field_typed_x")
    for generator in (positive, negative, dummy):
        polynomial = sp.Poly(generator**2 + 1, generator, modulus=3)
        derivative = polynomial.diff()
        assert derivative.gen is generator
        assert derivative.as_expr().free_symbols == {generator}
        assert next(iter(derivative.as_expr().free_symbols)) is generator
    with pytest.raises(UnificationFailed):
        sp.Poly(positive, positive, modulus=3) + sp.Poly(negative, negative, modulus=3)
    with pytest.raises((ValueError, NotImplementedError)):
        sp.Poly(negative, positive, modulus=3)


@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power", "cancellation"])
def test_original_unsupported_children_cannot_disappear_during_normalization(wrapper):
    x = sp.Symbol("x")
    invalid = sp.sin(x)
    if wrapper == "zero_product":
        source = sp.Mul(0, invalid, evaluate=False)
    elif wrapper == "zero_power":
        source = sp.Pow(invalid, 0, evaluate=False)
    else:
        source = sp.Add(invalid, sp.Mul(-1, invalid, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        sp.Poly(source, x, modulus=5)


@pytest.mark.parametrize("field", ["_expr", "_gens", "_domain", "_modulus", "_symmetric"])
def test_source_and_domain_mutation_during_checked_native_work_is_terminal(monkeypatch, field):
    x, y = sp.symbols("field_mutation_x field_mutation_y")
    polynomial = sp.Poly(x + 1, x, modulus=5)
    original = _native.prime_field_polynomial_native
    replacements = {"_expr": x + 2, "_gens": (y,), "_domain": "GF(7)", "_modulus": 7, "_symmetric": False}

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(polynomial, field, replacements[field])
        return result

    monkeypatch.setattr(_native, "prime_field_polynomial_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial + 1


def test_held_source_mutation_during_constructor_and_coefficient_container_mutation_refuse(monkeypatch):
    x = sp.Symbol("field_held_mutation_x")
    held = sp.Add(x, 1, evaluate=False)
    original = _native.prime_field_polynomial_native

    def mutate_held(*args, **kwargs):
        result = original(*args, **kwargs)
        held._args = (x, sp.Integer(2))
        return result

    monkeypatch.setattr(_native, "prime_field_polynomial_native", mutate_held)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.Poly(held, x, modulus=5)
    coefficients = [1, 2, 3]

    def mutate_list(*args, **kwargs):
        result = original(*args, **kwargs)
        coefficients[0] = 4
        return result

    monkeypatch.setattr(_native, "prime_field_polynomial_native", mutate_list)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.Poly.from_list(coefficients, x, modulus=5)


def test_mutation_during_lifting_and_stale_public_assumptions_refuse(monkeypatch):
    x = sp.Symbol("field_lift_mutation_x", positive=True)
    polynomial = sp.Poly(x + 1, x, modulus=5)
    original_lift = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original_lift(*args, **kwargs)
        polynomial._modulus = 7
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial.monic()
    monkeypatch.setattr(_finitefield, "_lift_native_result", original_lift)
    assumptions = dict(x._assumptions)
    try:
        x._assumptions["positive"] = False
        with pytest.raises(ValueError, match="identity"):
            sp.Poly(x + 1, x, modulus=5)
    finally:
        x._assumptions.clear()
        x._assumptions.update(assumptions)


def test_budget_and_verifier_failures_are_not_reclassified_or_retried(monkeypatch):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x + 1, x, modulus=5)
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args[4])
        raise ValueError("independent verifier budget exhausted")

    monkeypatch.setattr(_native, "prime_field_polynomial_native", refuse)
    with pytest.raises(ValueError, match="verifier budget"):
        polynomial**0
    assert attempts == ["pow"]


def test_only_typed_noninteger_coefficient_failures_are_mapped_to_coercion(monkeypatch):
    x = sp.Symbol("x")

    def refuse(*args, **kwargs):
        raise _native.PrimeFieldCoefficientError("final coefficient is not integral")

    monkeypatch.setattr(_native, "prime_field_polynomial_native", refuse)
    with pytest.raises(CoercionFailed):
        sp.Poly(x + 1, x, modulus=5)


def test_resultants_discriminants_and_scalar_domain_lifts():
    x = sp.Symbol("x")
    left = sp.Poly(2*x**2 + 3*x + 1, x, modulus=5)
    right = sp.Poly(x + 2, x, modulus=5)
    assert left.resultant(right) == -2
    assert left.discriminant() == 1
    assert sp.resultant(left, right) == -2
    assert sp.discriminant(left) == 1
    assert sp.resultant(2*x**2 + 3*x + 1, x + 2, x, modulus=5) == -2
    assert sp.discriminant(2*x**2 + 3*x + 1, x, domain="GF(5)") == 1
    unsigned = sp.Poly(2*x**2 + 3*x + 1, x, modulus=5, symmetric=False)
    assert unsigned.resultant(right) == 3
    for value in (0, 1, 3):
        assert sp.Poly(value, x, modulus=5).discriminant() == 0
    assert sp.Poly(3*x + 2, x, modulus=5).discriminant() == 1
    assert sp.Poly((x + 1)**2, x, modulus=5).discriminant() == 0
    assert sp.Poly(x**5 + 1, x, modulus=5).discriminant() == 0
    with pytest.raises(UnificationFailed):
        left.resultant(sp.Poly(x + 2, x, modulus=3))
    with pytest.raises(UnificationFailed):
        sp.resultant(sp.Poly(x + 2, x, domain="QQ"), right)


@pytest.mark.parametrize("prime, coefficients, resultant, discriminant", [
    (5, [2, 0, 1, 0, 0, 1], -2, 1),
    (3, [2, 1, 0, 1], 1, -1),
])
def test_discriminant_specializes_the_integer_polynomial_when_derivative_degree_drops(
        prime, coefficients, resultant, discriminant):
    # Deliberate, mathematically checked discrepancy from pinned SymPy1.14:
    # upstream loses the leading-coefficient power when differentiation drops
    # degree in characteristic p, returning respectively -1 and +1 here.
    # These expected discriminants are reductions of the integer discriminant,
    # independently certified by the formally padded Sylvester determinant.
    x = sp.Symbol("x")
    polynomial = sp.Poly.from_list(coefficients, x, modulus=prime)
    assert polynomial.resultant(polynomial.diff()) == resultant
    assert polynomial.discriminant() == discriminant


def test_scalar_results_and_equality_guard_every_original_operand_through_publication(monkeypatch):
    x = sp.Symbol("field_scalar_mutation_x")
    left = sp.Poly(x**2 + 1, x, modulus=5)
    right = sp.Poly(x + 1, x, modulus=5)
    original = _native.prime_field_polynomial_native

    def mutate_scalar(*args, **kwargs):
        result = original(*args, **kwargs)
        right._modulus = 7
        return result

    monkeypatch.setattr(_native, "prime_field_polynomial_native", mutate_scalar)
    with pytest.raises(RuntimeError, match="mutated"):
        left.resultant(right)
    monkeypatch.setattr(_native, "prime_field_polynomial_native", original)
    right = sp.Poly(x**2 + 1, x, modulus=5)
    projections = []

    def mutate_earlier_operand(*args, **kwargs):
        result = original(*args, **kwargs)
        projections.append(args[4])
        if len(projections) == 2:
            left._domain = "GF(7)"
        return result

    monkeypatch.setattr(_native, "prime_field_polynomial_native", mutate_earlier_operand)
    with pytest.raises(RuntimeError, match="mutated"):
        left == right


def test_factor_order_uses_canonical_residues_independent_of_display_policy():
    x = sp.Symbol("x")
    expression = (x - 1)*(x + 2)*(x + 1)*(x - 2)
    signed = sp.Poly(expression, x, modulus=7)
    unsigned = sp.Poly(expression, x, modulus=7, symmetric=False)
    assert [factor.all_coeffs() for factor, _ in signed.factor_list()[1]] == [
        [1, 1], [1, 2], [1, -2], [1, -1]]
    assert [factor.all_coeffs() for factor, _ in unsigned.factor_list()[1]] == [
        [1, 1], [1, 2], [1, 5], [1, 6]]


@pytest.mark.parametrize("control", ["modulus", "exponent", "derivative_order", "coefficient"])
def test_original_sources_are_retained_before_scalar_preflight(monkeypatch, control):
    x = sp.Symbol("field_scalar_preflight_x")
    held = sp.Add(x, 1, evaluate=False)
    polynomial = sp.Poly(held, x, modulus=5)
    original = _native.prime_field_source_preflight_native
    source = held if control == "modulus" else polynomial._expr
    before = source._value
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            calls.append(True)
            source._value = _native.py_integer(0)
        return result

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            if control == "modulus":
                sp.Poly(held, x, modulus=sp.Integer(5))
            elif control == "exponent":
                polynomial**sp.Integer(2)
            elif control == "derivative_order":
                polynomial.diff((x, sp.Integer(2)))
            else:
                # A coefficient is itself retained before another coefficient
                # can invoke the scalar preflight boundary.
                coefficients = [sp.Integer(2), sp.Integer(3)]
                source = coefficients[1]
                before = source._value
                sp.Poly.from_list(coefficients, x, modulus=5)
        assert calls == [True]
    finally:
        source._value = before


def test_modulus_and_generator_containers_remain_bound_to_the_whole_request(monkeypatch):
    x, y = sp.symbols("field_option_x field_option_y")
    prime = sp.Integer(5)
    original = _native.prime_field_source_preflight_native
    prime_value = prime._value

    def mutate_modulus(*args, **kwargs):
        result = original(*args, **kwargs)
        prime._value = _native.py_integer(7)
        return result

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate_modulus)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            sp.Poly(x + 1, x, modulus=prime)
    finally:
        prime._value = prime_value
    generators = [x]

    def mutate_generators(*args, **kwargs):
        result = original(*args, **kwargs)
        generators[0] = y
        return result

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate_generators)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.Poly(x + 1, generators, modulus=prime)


@pytest.mark.parametrize("operation", [sp.factor, sp.factor_list])
def test_factor_entry_retains_source_before_explicit_modulus_preflight(monkeypatch, operation):
    x = sp.Symbol("field_factor_preflight_x")
    polynomial = sp.Poly(x**2 + 1, x, modulus=5)
    source = polynomial._expr
    before = source._value
    original = _native.prime_field_source_preflight_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source._value = _native.py_integer(0)
        return result

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            operation(polynomial, modulus=sp.Integer(5))
    finally:
        source._value = before


def test_unsupported_field_options_refuse_before_native_generation(monkeypatch):
    x = sp.Symbol("x")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported options reached the native arithmetic kernel")

    monkeypatch.setattr(_native, "prime_field_polynomial_native", forbidden)
    with pytest.raises(NotImplementedError, match="extension"):
        sp.Poly(x + 1, x, modulus=5, extension=True)
    with pytest.raises(NotImplementedError, match="includePRS"):
        sp.resultant(x + 1, x**2 + 1, x, modulus=5, includePRS=True)
    with pytest.raises(TypeError, match="polys"):
        sp.div(x**2 + 1, x + 1, x, modulus=5, polys=1)
