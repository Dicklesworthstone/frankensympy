"""Finite-field radicals preserve all geometric points, including inseparability."""

import pytest

import sympy as s
from sympy.core import _native, _native_expr
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import GroebnerBasis


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_frobenius_thickening_recovers_all_points_and_coordinates(prime):
    x, y = s.symbols("finite_radical_x finite_radical_y")
    original = s.groebner([(x-y)**prime, y**prime-y], x, y,
                          modulus=prime, order="grevlex")
    result = original.radical()
    expected = s.groebner([x-y, y**prime-y], x, y, modulus=prime)
    assert result == expected
    assert result.gens[0] is x and result.gens[1] is y
    assert result.domain == "GF(%d)" % prime and result.order == "lex"
    assert original.is_radical is False and result.is_radical is True
    assert result.radical() == result
    assert len(result.quotient_basis()) == prime


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_nested_frobenius_power_is_not_stopped_at_zero_derivative(prime):
    x = s.Symbol("finite_radical_nested_x")
    original = s.groebner([(x-1)**(prime*prime)], x, modulus=prime)
    result = original.radical()
    assert result == s.groebner([x-1], x, modulus=prime)
    assert original.is_radical is False and result.is_radical is True


def test_irreducible_factor_without_base_field_roots_remains_proper():
    x = s.Symbol("finite_radical_extension_x")
    original = s.groebner([(x*x+1)**3], x, modulus=3)
    result = original.radical()
    assert result.exprs == [x*x+1]
    assert result.is_radical is True and original.is_radical is False
    assert not result.contains(1)
    # Adding x**3-x would wrongly replace the geometric ideal by its F_3 points.
    assert not result.contains(x**3-x)
    assert len(result.quotient_basis()) == 2


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_independent_nilpotent_directions_are_all_removed(prime):
    x, y, z = s.symbols("finite_radical_nil_x finite_radical_nil_y finite_radical_nil_z")
    original = s.groebner([x*x, x*y, y**3, z**2], x, y, z, modulus=prime)
    result = original.radical()
    assert result.exprs == [x, y, z]
    assert not original.contains(x) and not original.contains(y) and not original.contains(z)
    assert len(result.quotient_basis()) == 1 and result.is_radical is True


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("polys", [False, True])
def test_field_coefficient_and_polynomial_output_policies_survive(symmetric, polys):
    x = s.Symbol("finite_radical_policy_x")
    original = s.groebner([(x+3)**2], x, modulus=5, order="grlex",
                          symmetric=symmetric, polys=polys)
    result = original.radical()
    assert result._modulus == 5 and result._symmetric is symmetric and result._polys is polys
    assert result.exprs == [x-2 if symmetric else x+3]
    assert result.gens[0] is x
    if polys:
        assert type(result[0]) is s.Poly and result[0].get_modulus() == 5
        assert result[0].gens[0] is x and result[0].as_expr() == result.exprs[0]
    else:
        assert result[0] == result.exprs[0]
    assert result.is_radical is True


def test_dummy_and_same_spelled_distinct_axes_remain_distinct():
    plain = s.Symbol("finite_radical_typed")
    real = s.Symbol("finite_radical_typed", real=True)
    dummy = s.Dummy("finite_radical_typed")
    original = s.groebner([plain**2, real**3, dummy**2], dummy, real, plain, modulus=3)
    result = original.radical()
    assert all(a is b for a, b in zip(result.gens, (dummy, real, plain)))
    assert all(a is b for a, b in zip(result.exprs, (dummy, real, plain)))


def test_current_generators_are_admitted_even_after_public_list_mutation():
    x, y = s.symbols("finite_radical_current_x finite_radical_current_y")
    original = s.groebner([x], x, y, modulus=3)
    original.exprs[:] = [x*x, y*y, s.Integer(0), 2*x*x]
    assert original.radical().exprs == [x, y]
    assert original.is_radical is False


@pytest.mark.parametrize("prime", [2, 5])
def test_unit_ideal_remains_radical(prime):
    x, y = s.symbols("finite_radical_unit_x finite_radical_unit_y")
    original = s.groebner([1], x, y, modulus=prime)
    assert original.radical() == original
    assert original.is_radical is True


def test_explicit_constant_fields_are_already_reduced():
    x = s.Symbol("finite_radical_constant_x")
    zero = s.groebner([x*x+1], x, modulus=3).eliminate(x)
    unit = s.groebner([1], x, modulus=3).eliminate(x)
    assert zero.radical() == zero and unit.radical() == unit
    assert zero.is_radical is True and unit.is_radical is True


@pytest.mark.parametrize("zero", [False, True])
def test_positive_dimensional_ideal_raises_instead_of_partial_radical(zero):
    x, y = s.symbols("finite_radical_dimension_x finite_radical_dimension_y")
    original = s.groebner([] if zero else [x*x], x, y, modulus=3)
    with pytest.raises(ValueError, match="zero-dimensional"):
        original.radical()
    with pytest.raises(ValueError, match="zero-dimensional"):
        _ = original.is_radical


@pytest.mark.parametrize("kind", ["pole", "held_cancel", "rational", "undeclared"])
def test_unit_ideal_does_not_hide_invalid_original_sources(kind):
    x, y = s.symbols("finite_radical_hidden_x finite_radical_hidden_y")
    original = s.groebner([1], x, modulus=3)
    bad = {
        "pole": s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
        "held_cancel": s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False),
        "rational": s.Rational(1, 2),
        "undeclared": y,
    }[kind]
    original.exprs.append(bad)
    with pytest.raises((ValueError, CoercionFailed)):
        original.radical()
    with pytest.raises((ValueError, CoercionFailed)):
        _ = original.is_radical


@pytest.mark.parametrize("change", ["list", "generators", "prime", "policy", "held", "native", "assumptions"])
def test_retained_source_mutants_are_rejected_after_native_computation(monkeypatch, change):
    x = s.Symbol("finite_radical_mutation_"+change)
    original = s.groebner([x*x], x, modulus=3)
    held = s.Pow(x, 2, evaluate=False)
    original.exprs[:] = [held]
    native = _native.prime_field_ideal_radical_native

    def mutate(*args):
        result = native(*args)
        if change == "list":
            original.exprs.append(s.Integer(1))
        elif change == "generators":
            original.gens = ()
        elif change == "prime":
            original._modulus = 5
        elif change == "policy":
            original._symmetric = False
        elif change == "held":
            held._args = (x, s.Integer(3))
        elif change == "native":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_radical_native", mutate)
    with pytest.raises((RuntimeError, ValueError), match="mutat|chang"):
        original.radical()


def test_final_lifting_still_guards_the_original_ideal(monkeypatch):
    from sympy.polys import _finitefield_ideals as ideals

    x = s.Symbol("finite_radical_lift_x")
    original = s.groebner([x*x], x, modulus=3)
    lift = ideals._lift_native_result

    def mutate(*args):
        result = lift(*args)
        original.exprs.append(s.Integer(1))
        return result

    monkeypatch.setattr(ideals, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        original.radical()


def test_operational_failures_never_retry_over_QQ(monkeypatch):
    x = s.Symbol("finite_radical_refusal_x")
    original = s.groebner([x*x], x, modulus=3)

    def exhausted(*args):
        raise ValueError("test modular resource exhausted")

    def forbidden(*args):
        pytest.fail("a modular ideal was retried over QQ")

    monkeypatch.setattr(_native, "prime_field_ideal_radical_native", exhausted)
    monkeypatch.setattr(_native, "ideal_radical_native", forbidden)
    with pytest.raises(ValueError, match="test modular resource exhausted"):
        original.radical()
    with pytest.raises(ValueError, match="test modular resource exhausted"):
        _ = original.is_radical
