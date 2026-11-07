"""Public modular ideal operations, checked against hand-derived identities.

These extension methods preserve arithmetic ideals, including multiplicity and
the coefficient field. They do not replace ideals with their finite point sets.
The tests require the actual native extension; there is no oracle fallback.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr
from sympy.polys.polyerrors import (
    CoercionFailed, DomainError, GeneratorsError, GeneratorsNeeded,
    UnificationFailed,
)
from sympy.polys.polytools import GroebnerBasis


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_nonleading_elimination_recovers_the_complete_cusp_ideal(prime):
    x, z, y = sp.symbols("modular_cusp_x modular_cusp_z modular_cusp_y")
    original = sp.groebner([x-y**2, z-y**3], x, z, y, modulus=prime, order="grlex")
    result = original.eliminate(y)
    expected = x**3 + z**2 if prime == 2 else x**3-z**2
    assert type(result) is GroebnerBasis
    assert result.exprs == [expected]
    assert result.gens == (x, z) and result.gens[0] is x and result.gens[1] is z
    assert result.domain == "GF(%d)" % prime and result.order == "lex"
    assert result.contains(x**3-z**2)
    assert not result.contains(x-z)
    assert all(not expression.has(y) for expression in result)


def test_elimination_preserves_the_complement_order_and_unused_coordinates():
    w, z, y, x = sp.symbols("modular_order_w modular_order_z modular_order_y modular_order_x")
    original = sp.groebner([x-y, z-y**2], w, z, y, x, modulus=5)
    result = original.eliminate([y])
    assert result.gens == (w, z, x) and result.exprs == [z-x**2]
    assert all(left is right for left, right in zip(result.gens, (w, z, x)))
    assert original.eliminate(y, x).gens == (w, z)
    assert original.eliminate(y, x).exprs == []
    unchanged = original.eliminate()
    assert unchanged.gens == original.gens and unchanged.order == "lex"
    assert all(unchanged.contains(expression) for expression in original.exprs)


@pytest.mark.parametrize("polys", [False, True])
def test_all_variable_elimination_is_a_usable_constant_field(polys):
    x = sp.Symbol("modular_constant_axis")
    # x**2+1 has no F_3-rational root, but generates a proper polynomial ideal.
    zero = sp.groebner([x**2+1], x, modulus=3, polys=polys).eliminate(x)
    unit = sp.groebner([1], x, modulus=3, polys=polys).eliminate(x)
    assert zero.gens == unit.gens == ()
    assert zero.exprs == [] and unit.exprs == [1]
    assert zero.domain == unit.domain == "GF(3)"
    assert zero.contains(0) and not zero.contains(1)
    assert unit.contains(7)
    assert zero.intersection(unit).exprs == []
    assert unit.intersection(unit).exprs == [1]
    assert zero.eliminate().exprs == [] and unit.eliminate([]).exprs == [1]
    assert zero.saturate(0).exprs == [1] and zero.saturate(2).exprs == []
    assert unit.saturate(0).exprs == [1]
    quotients, remainder = unit.reduce(7)
    if polys:
        assert type(unit[0]) is sp.Poly and unit[0].gens == ()
        assert unit[0].as_expr() == 1 and unit[0].get_modulus() == 3
        assert unit[0].terms() == [((), sp.Integer(1))]
        assert unit[0].degree_list() == () and unit[0].total_degree() == 0
        assert unit[0].nth() == 1 and unit[0].is_one
        assert unit[0] == 1 and unit[0] == sp.Integer(4) and unit[0] != x
        with pytest.raises(GeneratorsError):
            unit[0].diff()
        quotient, rest = unit[0].div(2)
        assert quotient.as_expr() == -1 and rest.as_expr() == 0
        assert quotient.gens == rest.gens == ()
        assert len(quotients) == 1 and quotients[0].as_expr() == 1
        assert type(remainder) is sp.Poly and remainder.gens == () and remainder.is_zero
    else:
        assert quotients == [1] and remainder == 0
    # This new internal closure does not invent an inferred public coordinate.
    with pytest.raises(GeneratorsNeeded):
        sp.Poly(1, modulus=3)
    with pytest.raises(GeneratorsNeeded):
        sp.groebner([1], modulus=3)


@pytest.mark.parametrize("prime", [2, 5])
def test_intersection_retains_all_generators_and_multiplicities(prime):
    x, y = sp.symbols("modular_intersection_x modular_intersection_y")
    first = sp.groebner([x**2, y], x, y, modulus=prime)
    second = sp.groebner([x, y**2], y, x, modulus=prime, order="grevlex")
    result = first.intersection(second)
    assert result.gens == (x, y) and result.order == "lex"
    assert set(result.exprs) == {x**2, x*y, y**2}
    assert all(first.contains(expression) and second.contains(expression)
               for expression in result.exprs)
    assert not result.contains(x) and not result.contains(y)
    assert sp.groebner([x], x, y, modulus=prime).intersection(
        sp.groebner([y], y, x, modulus=prime)).exprs == [x*y]


def test_zero_and_unit_ideal_laws_keep_the_explicit_field():
    x, y = sp.symbols("modular_law_x modular_law_y")
    zero = sp.groebner([], x, y, modulus=5)
    unit = sp.groebner([1], x, y, modulus=5)
    ideal = sp.groebner([x*y-1], x, y, modulus=5)
    assert zero.intersection(ideal).exprs == ideal.intersection(zero).exprs == []
    assert unit.intersection(ideal).exprs == ideal.intersection(unit).exprs == ideal.exprs
    assert ideal.saturate(1).exprs == ideal.saturate(2).exprs == ideal.exprs
    assert ideal.saturate(5).exprs == [1]
    assert zero.saturate(x).exprs == [] and zero.saturate(0).exprs == [1]


@pytest.mark.parametrize("prime", [2, 5])
def test_saturation_removes_arbitrary_factor_multiplicity_without_point_filtering(prime):
    x, y = sp.symbols("modular_saturation_x modular_saturation_y")
    original = sp.groebner([x**3*y], x, y, modulus=prime)
    result = original.saturate(x)
    assert result.exprs == [y] and result.gens == (x, y)
    assert result.contains(y) and not original.contains(y)
    assert not result.contains(x) and not result.contains(1)
    assert original.saturate(x*y).exprs == [1]
    assert original.exprs == [x**3*y]
    # In characteristic two, x**2+1=(x+1)**2; saturation removes both factors.
    if prime == 2:
        assert sp.groebner([x**2+1], x, modulus=2).saturate(x+1).exprs == [1]


def test_an_ideal_without_rational_points_is_not_replaced_with_the_unit_ideal():
    x = sp.Symbol("modular_algebraic_point_x")
    original = sp.groebner([x**2+1], x, modulus=3)
    assert original.saturate(x).exprs == [x**2+1]
    assert original.intersection(sp.groebner([1], x, modulus=3)).exprs == [x**2+1]
    assert original.eliminate(x).exprs == []
    assert not original.contains(x**3-x)


def test_output_policies_survive_a_chain_of_all_three_operations():
    x, y, z = sp.symbols("modular_policy_x modular_policy_y modular_policy_z")
    original = sp.groebner([z*(x-y)], x, y, z, modulus=5, order="grlex",
                          polys=True, symmetric=False)
    factor = sp.Poly(z, z, x, y, modulus=5)
    saturated = original.saturate(factor)
    eliminated = saturated.eliminate(z)
    result = eliminated.intersection(sp.groebner([1], y, x, modulus=5, polys=False))
    assert result.exprs == [x+4*y] and result.gens == (x, y)
    assert result.domain == "GF(5)" and result.order == "lex"
    assert all(type(poly) is sp.Poly and poly.get_modulus() == 5 for poly in result)
    assert result[0].as_expr() == x+4*y
    assert result[0].gens[0] is x and result[0].gens[1] is y
    assert result.reduce(x)[1] == sp.Poly(y, x, y, modulus=5, symmetric=False)
    expected = sp.groebner([x-y], x, y, modulus=5, polys=True, symmetric=False)
    assert result == expected and hash(result) == hash(expected)


def test_typed_same_printed_symbols_and_dummy_owners_survive_elimination_and_saturation():
    plain = sp.Symbol("modular_shared_axis")
    assumed = sp.Symbol("modular_shared_axis", real=True)
    dummy = sp.Dummy("modular_shared_axis")
    original = sp.groebner([dummy-assumed, plain-assumed**2], dummy, assumed, plain, modulus=5)
    result = original.eliminate(assumed)
    assert result.gens[0] is dummy and result.gens[1] is plain
    assert result.exprs == [dummy**2-plain]
    assert all(any(symbol is owner for symbol in result.exprs[0].free_symbols)
               for owner in (dummy, plain))
    product = sp.groebner([plain*assumed*dummy], dummy, assumed, plain, modulus=5)
    saturated = product.saturate(plain)
    assert saturated.exprs == [assumed*dummy]
    assert all(left is right for left, right in zip(saturated.gens, product.gens))
    with pytest.raises(ValueError, match="original typed ring"):
        sp.groebner([plain], plain, modulus=5).eliminate(assumed)


def test_auxiliary_parameter_names_do_not_capture_original_axes():
    parameter = sp.Symbol("_fsym_prime_field_ideal_parameter_0")
    dummy = sp.Dummy("_fsym_prime_field_ideal_parameter_1")
    x = sp.Symbol("modular_auxiliary_x")
    first = sp.groebner([parameter*x], parameter, dummy, x, modulus=5)
    second = sp.groebner([dummy], dummy, x, parameter, modulus=5)
    intersection = first.intersection(second)
    assert intersection.exprs == [parameter*dummy*x]
    result = intersection.saturate(parameter*x)
    assert result.exprs == [dummy]
    assert result.gens == (parameter, dummy, x)
    assert next(iter(result.exprs[0].free_symbols)) is dummy


def test_domain_and_typed_ring_mismatches_are_refused_without_conversion():
    x = sp.Symbol("modular_mismatch_x")
    same_printed = sp.Symbol("modular_mismatch_x", real=True)
    y = sp.Symbol("modular_mismatch_y")
    first = sp.groebner([x], x, modulus=5)
    for other in (sp.groebner([x], x, modulus=7),
                  sp.groebner([x], x, domain="QQ"),
                  sp.groebner([same_printed], same_printed, modulus=5)):
        with pytest.raises(UnificationFailed):
            first.intersection(other)
        with pytest.raises(UnificationFailed):
            other.intersection(first)
    for factor in (sp.Poly(x, x, modulus=7), sp.Poly(0, x, y, modulus=5),
                   sp.Poly(1, same_printed, modulus=5)):
        with pytest.raises(UnificationFailed):
            first.saturate(factor)
    assert first.saturate(sp.Poly(x, x, domain="ZZ")).exprs == [1]
    with pytest.raises(CoercionFailed):
        first.saturate(sp.Poly(sp.Rational(1, 2), x, domain="QQ"))
    with pytest.raises(ValueError, match="distinct"):
        first.eliminate(x, x)


def test_unused_poly_axes_must_match_their_immutable_native_identity(monkeypatch):
    target = sp.Symbol("modular_unused_identity")
    foreign = sp.Symbol("modular_unused_identity", real=True)
    basis = sp.groebner([target], target, modulus=5)
    factor = sp.Poly(0, foreign, modulus=5)
    # Changing only current declarations cannot turn the retained native atom
    # into the target coordinate, even when a zero source never uses that atom.
    monkeypatch.setattr(foreign, "_assumptions", dict(target._assumptions))
    with pytest.raises(ValueError, match="identity"):
        basis.saturate(factor)
    assert basis.saturate(sp.Poly(0, target, modulus=5)).exprs == [1]


def test_current_sources_need_not_already_be_a_groebner_basis():
    x, y = sp.symbols("modular_current_x modular_current_y")
    first = sp.groebner([x], x, y, modulus=5)
    first.exprs[:] = [2*x*y, 0, x*y]
    second = sp.groebner([y], y, x, modulus=5)
    second.exprs[:] = [3*y**2, y**2]
    assert first.saturate(x).exprs == [y]
    assert first.intersection(second).exprs == [x*y**2]
    assert first.eliminate(x).exprs == []


@pytest.mark.parametrize("bad_kind", ["reciprocal", "rational", "float", "unknown"])
def test_original_invalid_sources_survive_zero_and_unit_shortcuts(bad_kind):
    x = sp.Symbol("modular_original_" + bad_kind)
    bad = {"reciprocal": sp.Pow(x, -1, evaluate=False),
           "rational": sp.Rational(1, 2), "float": sp.Float("1.0"),
           "unknown": sp.sin(x)}[bad_kind]
    hidden = sp.Mul(0, bad, evaluate=False)
    first = sp.groebner([1], x, modulus=5)
    other = sp.groebner([], x, modulus=5)
    first.exprs[:] = [1, hidden]
    for operation in (lambda: first.eliminate(x), lambda: first.intersection(other),
                      lambda: other.intersection(first), lambda: first.saturate(0),
                      lambda: other.saturate(hidden)):
        with pytest.raises((ValueError, CoercionFailed)):
            operation()


@pytest.mark.parametrize("field", ["domain", "_modulus", "_symmetric", "_polys"])
def test_inconsistent_basis_metadata_never_reaches_either_arithmetic_engine(monkeypatch, field):
    x = sp.Symbol("modular_metadata_" + field)
    basis = sp.groebner([x], x, modulus=5)
    other = sp.groebner([1], x, modulus=5)
    setattr(basis, field, {"domain": "QQ", "_modulus": 7,
                           "_symmetric": 1, "_polys": 1}[field])

    def forbidden(*args, **kwargs):
        raise AssertionError("inconsistent metadata reached native ideal arithmetic")

    for name in ("prime_field_ideal_eliminate_native", "prime_field_ideal_intersection_native",
                 "prime_field_ideal_saturation_native", "ideal_eliminate_native",
                 "ideal_intersection_native", "ideal_saturation_native"):
        monkeypatch.setattr(_native, name, forbidden)
    for operation in (lambda: basis.eliminate(x), lambda: basis.intersection(other),
                      lambda: other.intersection(basis), lambda: basis.saturate(x)):
        with pytest.raises(DomainError):
            operation()


@pytest.mark.parametrize("change", ["left_list", "right_list", "right_gens", "domain",
                                    "held_children", "native_value", "assumptions"])
def test_intersection_retains_both_original_operands_through_native_verification(monkeypatch, change):
    x, y = sp.symbols("modular_mutation_" + change + "_x modular_mutation_" + change + "_y")
    held = sp.Add(x, y, evaluate=False)
    first = sp.groebner([x], x, y, modulus=5)
    first.exprs[:] = [held]
    second = sp.groebner([1], y, x, modulus=5)
    original = _native.prime_field_ideal_intersection_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "left_list":
            first.exprs.append(x)
        elif change == "right_list":
            second.exprs.append(x)
        elif change == "right_gens":
            second.gens = (x, y)
        elif change == "domain":
            second.domain = "GF(7)"
        elif change == "held_children":
            held._args = (x, x)
        elif change == "native_value":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_intersection_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.intersection(second)


def test_right_operand_is_captured_before_left_source_preflight(monkeypatch):
    x = sp.Symbol("modular_preflight_x")
    first = sp.groebner([x], x, modulus=5)
    second = sp.groebner([1], x, modulus=5)
    original = _native.ideal_source_preflight_native
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            calls.append(1)
            second.exprs.append(x)
        return result

    monkeypatch.setattr(_native, "ideal_source_preflight_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.intersection(second)
    assert calls == [1]


def test_factor_poly_metadata_and_elimination_controls_are_retained(monkeypatch):
    x, y = sp.symbols("modular_control_x modular_control_y")
    basis = sp.groebner([x*y], x, y, modulus=5)
    factor = sp.Poly(x, x, y, modulus=5)
    original_saturate = _native.prime_field_ideal_saturation_native

    def change_factor(*args, **kwargs):
        result = original_saturate(*args, **kwargs)
        factor._modulus = 7
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_saturation_native", change_factor)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.saturate(factor)
    axes = [x]
    original_eliminate = _native.prime_field_ideal_eliminate_native

    def change_axes(*args, **kwargs):
        result = original_eliminate(*args, **kwargs)
        axes[0] = y
        return result

    monkeypatch.setattr(_native, "prime_field_ideal_eliminate_native", change_axes)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.eliminate(axes)


def test_output_lifting_cannot_publish_after_a_source_change(monkeypatch):
    from sympy.polys import _finitefield_ideals

    x, y = sp.symbols("modular_lift_x modular_lift_y")
    first = sp.groebner([x*y], x, y, modulus=5)
    original = _finitefield_ideals._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        first.exprs[:] = [x]
        return result

    monkeypatch.setattr(_finitefield_ideals, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        first.saturate(x)


@pytest.mark.parametrize("operation", ["eliminate", "intersection", "saturation"])
@pytest.mark.parametrize("failure", [ValueError, MemoryError])
def test_operational_failure_is_terminal_and_does_not_restart_over_qq(monkeypatch, operation, failure):
    x = sp.Symbol("modular_terminal_" + operation)
    basis = sp.groebner([x], x, modulus=5)
    other = sp.groebner([1], x, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure("prime-field verifier budget exhausted or cancelled")

    def forbidden(*args, **kwargs):
        raise AssertionError("a modular refusal restarted in rational arithmetic")

    monkeypatch.setattr(_native, "prime_field_ideal_" + operation + "_native", refuse)
    for name in ("ideal_eliminate_native", "ideal_intersection_native", "ideal_saturation_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(failure, match="budget exhausted or cancelled"):
        if operation == "eliminate":
            basis.eliminate(x)
        elif operation == "intersection":
            basis.intersection(other)
        else:
            basis.saturate(x)
    assert calls == [1]


def test_closed_operand_and_axis_admission_does_not_invoke_user_hooks():
    x = sp.Symbol("modular_no_hook_x")
    basis = sp.groebner([x], x, modulus=5)
    calls = []

    class HostileAxes(list):
        def __iter__(self):
            calls.append("axis iteration")
            raise AssertionError("untrusted axis iterator")

    class HostileFactor:
        def _sympy_(self):
            calls.append("factor coercion")
            raise AssertionError("untrusted factor coercion")

    class HostileBasis(GroebnerBasis):
        def __getattribute__(self, name):
            calls.append("basis attribute")
            raise AssertionError("untrusted basis attribute")

    with pytest.raises(GeneratorsError):
        basis.eliminate(HostileAxes([x]))
    with pytest.raises(ValueError):
        basis.saturate(HostileFactor())
    with pytest.raises(TypeError):
        basis.intersection(object.__new__(HostileBasis))
    assert calls == []


def test_ambient_variable_and_source_count_bounds_are_explicit():
    axes = sp.symbols("modular_limit_x0:8")
    original = sp.groebner([axes[0]], *axes, modulus=5)
    # Elimination needs no auxiliary variable and can use all eight coordinates.
    assert original.eliminate(axes[-1]).gens == axes[:-1]
    with pytest.raises(ValueError):
        original.saturate(axes[0])
    with pytest.raises(ValueError):
        original.intersection(original)
    x = axes[0]
    first = sp.groebner([x], x, modulus=5)
    first.exprs[:] = [x] * 32
    with pytest.raises(ValueError, match="31"):
        first.saturate(x)
    second = sp.groebner([1], x, modulus=5)
    with pytest.raises(ValueError, match="32"):
        first.intersection(second)
