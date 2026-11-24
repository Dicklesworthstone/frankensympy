"""Complete finite-field square-free groups through the native extension.

These fixtures assert a bounded multivariate extension; upstream SymPy's
multivariate GF(p) square-free refusal is not treated as an oracle result.
"""

import pytest
import sympy as sp

from sympy.core import _native
from sympy.polys import _finitefield
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, UnificationFailed


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_complete_groups_reconstruct_original_and_keep_the_field(prime, cls):
    x, y = sp.symbols("sqf_field_x sqf_field_y")
    source = cls((prime - 1)*(x + y + 1)**2*(y + 1)**3, x, y, modulus=prime)
    unit, groups = source.sqf_list()
    assert unit == (1 if prime == 2 else -1)
    assert groups == [
        (cls(x + y + 1, x, y, modulus=prime), 2),
        (cls(y + 1, x, y, modulus=prime), 3),
    ]
    reconstructed = cls(unit, x, y, modulus=prime)
    for factor, multiplicity in groups:
        assert type(factor) is cls
        assert factor.gens[0] is x and factor.gens[1] is y
        assert factor.get_modulus() == prime
        reconstructed *= factor**multiplicity
    assert reconstructed == source
    assert source.sqf_part() == cls((x + y + 1)*(y + 1), x, y, modulus=prime)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_frobenius_descent_recovers_two_distinct_inseparable_layers(prime):
    x, y = sp.symbols("sqf_frobenius_x sqf_frobenius_y")
    source = sp.Poly((x + y + 1)**(prime*prime)*(y + 1)**prime,
                     x, y, modulus=prime)
    assert source.diff(x).is_zero
    assert source.diff(y).is_zero
    assert source.sqf_list() == (1, [
        (sp.Poly(y + 1, x, y, modulus=prime), prime),
        (sp.Poly(x + y + 1, x, y, modulus=prime), prime*prime),
    ])
    assert source.sqf_part() == sp.Poly((x + y + 1)*(y + 1), x, y, modulus=prime)


def test_maximal_binary_frobenius_depth_does_not_lose_multiplicity():
    x, y = sp.symbols("sqf_depth_x sqf_depth_y")
    source = sp.Poly((x + y + 1)**64, x, y, modulus=2)
    assert source.sqf_list() == (1, [(sp.Poly(x + y + 1, x, y, modulus=2), 64)])


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_a_zero_first_derivative_does_not_skip_other_axes(prime):
    x, y = sp.symbols("sqf_partial_x sqf_partial_y")
    factor = x**prime + y
    source = sp.Poly(factor**2, x, y, modulus=prime)
    assert source.sqf_list() == (1, [(sp.Poly(factor, x, y, modulus=prime), 2)])


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("symmetric", [False, True])
def test_assumptions_dummy_identity_and_lift_policy_survive_publication(cls, symmetric):
    x = sp.Symbol("square_free_axis", positive=True)
    y = sp.Dummy("square_free_axis")
    source = cls(4*(x + 4*y)**3, x, y, modulus=5, symmetric=symmetric)
    unit, groups = source.sqf_list()
    expected = -1 if symmetric else 4
    assert unit == expected
    factor, multiplicity = groups[0]
    assert multiplicity == 3 and type(factor) is cls
    assert factor.gens[0] is x and factor.gens[1] is y
    assert dict(factor.terms())[(0, 1)] == expected
    assert factor.get_modulus() == 5
    assert source.sqf_part() == cls(x + 4*y, x, y, modulus=5, symmetric=symmetric)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 9])
def test_zero_and_constant_public_field_conventions(value):
    x, y = sp.symbols("sqf_constant_x sqf_constant_y")
    source = sp.Poly(value, x, y, modulus=3)
    expected = value % 3
    if expected > 1:
        expected -= 3
    assert source.sqf_list() == (expected, [])
    # This is the existing pinned GF(p) public convention, including zero.
    assert source.sqf_part().is_one
    assert sp.sqf_part(source).is_one


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_module_entries_support_original_expressions_and_explicit_fields(prime):
    x, y = sp.symbols("sqf_module_x sqf_module_y")
    expression = (x + y + 1)**2*(y + 1)**3
    source = sp.Poly(expression, x, y, modulus=prime)
    coefficient, groups = sp.sqf_list(expression, x, y, modulus=prime)
    assert coefficient == 1
    assert [(sp.Poly(factor, x, y, modulus=prime), count)
            for factor, count in groups] == source.sqf_list()[1]
    assert sp.sqf_list(source, x, y) == source.sqf_list()
    assert sp.sqf_list(expression, (x, y), domain="GF(%d)" % prime, polys=True) == source.sqf_list()
    assert sp.Poly(sp.sqf_part(expression, x, y, modulus=prime),
                   x, y, modulus=prime) == source.sqf_part()
    assert sp.Poly(sp.sqf(expression, x, y, modulus=prime),
                   x, y, modulus=prime) == source
    assert sp.sqf(source) == source
    assert type(sp.sqf(source)) is sp.Poly
    assert not isinstance(sp.sqf_part(source, polys=False), sp.Poly)


def test_equal_multiplicities_are_grouped_without_an_irreducibility_claim():
    x, y = sp.symbols("sqf_group_x sqf_group_y")
    factor = (x + y)*(x + 1)*(y + 1)
    source = sp.Poly(factor**3, x, y, modulus=2)
    assert source.sqf_list() == (1, [(sp.Poly(factor, x, y, modulus=2), 3)])
    with pytest.raises(NotImplementedError, match="irreducible"):
        source.factor_list()


def test_new_module_univariate_path_agrees_with_existing_poly_conventions():
    x = sp.Symbol("sqf_univariate_compatibility")
    for expression in (0, 2, (x + 1)**6*(x*x + 1)**3):
        source = sp.Poly(expression, x, modulus=3)
        unit, groups = sp.sqf_list(expression, x, modulus=3, polys=True)
        assert (unit, groups) == source.sqf_list()
        assert sp.sqf_part(expression, x, modulus=3, polys=True) == source.sqf_part()


@pytest.mark.parametrize("operation", ["sqf_list", "sqf_part"])
def test_original_held_poles_cannot_disappear_inside_zero_terms(operation):
    x, y = sp.symbols("sqf_pole_x sqf_pole_y")
    source = sp.Poly(x + y, x, y, modulus=3)
    pole = sp.Pow(sp.Integer(0), sp.Integer(-1), evaluate=False)
    source._expr = sp.Add(x, y, sp.Mul(0, pole, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError, CoercionFailed)):
        getattr(source, operation)()


def test_native_interface_substitution_is_refused_before_execution(monkeypatch):
    x, y = sp.symbols("sqf_interface_x sqf_interface_y")
    source = sp.Poly((x + y)**2, x, y, modulus=3)
    calls = []

    def replacement(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a replacement checker must not execute")

    monkeypatch.setattr(_native, "prime_field_multivariate_square_free_native", replacement)
    with pytest.raises(NotImplementedError, match="original native interface"):
        source.sqf_list()
    assert calls == []


def test_poly_source_mutation_during_publication_is_detected(monkeypatch):
    x, y = sp.symbols("sqf_mutation_x sqf_mutation_y")
    source = sp.Poly((x + y)**2, x, y, modulus=3)
    original = _finitefield._lift_native_result
    mutated = False

    def mutate(raw, *context):
        nonlocal mutated
        result = original(raw, *context)
        if not mutated:
            mutated = True
            source._expr = x + 1
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.sqf_list()


def test_changed_declared_generator_order_is_not_silently_inherited():
    x, y = sp.symbols("sqf_order_x sqf_order_y")
    source = sp.Poly((x + y)**2, x, y, modulus=3)
    with pytest.raises(UnificationFailed):
        sp.sqf_list(source, y, x)
    with pytest.raises(GeneratorsError):
        sp.sqf_list(source, x, y, gens=(x, y))
    with pytest.raises(TypeError):
        sp.sqf(source, polys=True)


def test_active_callback_cannot_restart_the_intrinsic_polynomial_budget():
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    x, y = sp.symbols("sqf_callback_x sqf_callback_y")
    source = sp.Poly((x + y)**2, x, y, modulus=3)
    session = SupervisedCallbackSession(max_nested_steps=20_000_000)
    with supervised_callback_session(session):
        with pytest.raises(NotImplementedError, match="supervised callback lane"):
            source.sqf_list()


def test_module_entries_preserve_the_existing_degree64_univariate_lane():
    x = sp.Symbol("sqf_degree64_univariate")
    source = sp.Poly(x**64, x, modulus=65521)
    assert sp.sqf_list(source) == source.sqf_list()
    assert sp.sqf_part(source) == source.sqf_part()
    assert sp.sqf_list(x**64, x, modulus=65521, polys=True) == source.sqf_list()


def test_explicit_field_option_cannot_erase_an_original_unused_axis():
    x, y = sp.symbols("sqf_original_x sqf_original_unused")
    source = sp.Poly(x**2, x, y, modulus=5)
    y._assumptions["commutative"] = False
    with pytest.raises((GeneratorsError, UnificationFailed)):
        sp.sqf_list(source, x, modulus=5)
