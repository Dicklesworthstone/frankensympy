"""Checked ideal operations are complete over the explicitly declared QQ ring."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys.polytools import GroebnerBasis


def test_elimination_of_noninitial_variable_recovers_the_complete_cusp_ideal():
    x, z, y = s.symbols("ideal_x ideal_z ideal_y")
    basis = s.groebner([x-y*y, z-y**3], x, z, y, domain="QQ", order="grlex")
    eliminated = basis.eliminate(y)
    assert eliminated.gens == (x, z)
    assert eliminated.domain == "QQ" and eliminated.order == "lex"
    assert eliminated.exprs == [x**3-z*z]
    assert eliminated.contains(x**3-z*z)
    assert not eliminated.contains(x-z)
    assert all(not expression.has(y) for expression in eliminated)


def test_elimination_preserves_retained_order_and_unconstrained_parameters():
    x, y, z, w = s.symbols("retained_x retained_y retained_z retained_w")
    basis = s.groebner([x-y, z-y*y], w, z, y, x, domain="QQ")
    eliminated = basis.eliminate(y)
    assert eliminated.gens == (w, z, x)
    assert eliminated.exprs == [z-x*x]
    assert basis.eliminate([y, x]).gens == (w, z)
    assert basis.eliminate(y, x).exprs == []
    unchanged = basis.eliminate()
    assert unchanged.gens == basis.gens
    assert all(unchanged.contains(expression) for expression in basis)


def test_eliminating_all_generators_preserves_zero_and_unit_constant_ideals():
    x = s.Symbol("constant_ideal_x")
    proper = s.groebner([x*x-2], x, domain="QQ").eliminate(x)
    unit = s.groebner([x, x-1], x, domain="QQ").eliminate(x)
    assert proper.gens == unit.gens == ()
    assert proper.exprs == [] and unit.exprs == [1]
    assert proper.contains(s.Integer(0))
    assert not proper.contains(s.Integer(1))
    assert unit.contains(s.Rational(2, 3))


def test_intersection_is_not_the_sum_of_the_two_ideals():
    x, y = s.symbols("intersection_x intersection_y")
    first = s.groebner([x], x, y, domain="QQ")
    second = s.groebner([y], y, x, domain="QQ", order="grevlex")
    result = first.intersection(second)
    assert result.gens == (x, y)
    assert result.exprs == [x*y]
    assert result.contains(x*y)
    assert not result.contains(x)
    assert not result.contains(y)
    assert second.intersection(first).exprs == [x*y]


def test_intersection_recovers_all_generators_of_a_nonprincipal_ideal():
    x, y = s.symbols("nonprincipal_x nonprincipal_y")
    first = s.groebner([x*x, y], x, y, domain="QQ")
    second = s.groebner([x, y*y], x, y, domain="QQ")
    result = first.intersection(second)
    assert set(result.exprs) == {x*x, x*y, y*y}
    for expression in result:
        assert first.contains(expression) and second.contains(expression)
    assert not result.contains(x) and not result.contains(y)


def test_zero_and_unit_intersection_laws_use_explicit_ambient_rings():
    x, y = s.symbols("zero_unit_x zero_unit_y")
    zero = s.groebner([], x, y, domain="QQ")
    unit = s.groebner([1], x, y, domain="QQ")
    ideal = s.groebner([x*y-1], x, y, domain="QQ")
    assert zero.intersection(ideal).exprs == []
    assert ideal.intersection(zero).exprs == []
    assert unit.intersection(ideal).exprs == ideal.exprs
    assert ideal.intersection(unit).exprs == ideal.exprs


def test_finite_quotient_preserves_multiplicity_instead_of_saturating():
    x, y = s.symbols("quotient_multiplicity_x quotient_multiplicity_y")
    ideal = s.groebner([x**3*y], x, y, domain="QQ")
    divisor = s.groebner([x], x, y, domain="QQ")
    result = ideal.quotient(divisor)
    assert result.exprs == [x*x*y]
    assert result.domain == "QQ" and result.order == "lex"
    assert ideal.saturate(x).exprs == [y]
    assert not result.contains(y)
    assert ideal.exprs == [x**3*y] and divisor.exprs == [x]


def test_nonmonomial_principal_quotient_returns_every_generator():
    x, y, z = s.symbols("quotient_linear_x quotient_linear_y quotient_linear_z")
    ideal = s.groebner([(x+y)**2, (x+y)*z], x, y, z, domain="QQ", order="grevlex")
    divisor = s.groebner([x+y], z, y, x, domain="QQ")
    result = ideal.quotient(divisor)
    assert result.gens == (x, y, z)
    assert result.exprs == [x+y, z]
    assert all(ideal.contains(expression*(x+y)) for expression in result)
    assert not result.contains(1)


def test_multi_generator_quotient_requires_all_products_not_any_one():
    x, y = s.symbols("quotient_all_x quotient_all_y")
    ideal = s.groebner([x*x, y*y], x, y, domain="QQ")
    divisor = s.groebner([y, x], y, x, domain="QQ", order="grlex")
    result = ideal.quotient(divisor)
    assert set(result.exprs) == {x*x, x*y, y*y}
    assert result.contains(x*y)
    assert not result.contains(x) and not result.contains(y)
    assert s.groebner([x*y], x, y, domain="QQ").quotient(divisor).exprs == [x*y]
    for expression in result:
        assert ideal.contains(expression*x) and ideal.contains(expression*y)


def test_quotient_zero_unit_and_constant_ring_laws_are_complete():
    x, y = s.symbols("quotient_edge_x quotient_edge_y")
    zero = s.groebner([], x, y, domain="QQ")
    unit = s.groebner([1], x, y, domain="QQ")
    ideal = s.groebner([x*y-1], x, y, domain="QQ")
    assert ideal.quotient(zero).exprs == [1]
    assert zero.quotient(zero).exprs == [1]
    assert zero.quotient(ideal).exprs == []
    assert ideal.quotient(unit).exprs == ideal.exprs
    assert unit.quotient(ideal).exprs == [1]
    assert ideal.quotient(ideal).exprs == [1]
    constant_zero = GroebnerBasis([], (), "QQ", "lex")
    constant_unit = GroebnerBasis([s.Rational(2, 3)], (), "QQ", "lex")
    assert constant_zero.quotient(constant_zero).exprs == [1]
    assert constant_zero.quotient(constant_unit).exprs == []
    assert constant_unit.quotient(constant_unit).exprs == [1]


def test_quotient_admits_current_generators_and_explicit_QQ_extension():
    x, y = s.symbols("quotient_current_x quotient_current_y")
    ideal = s.groebner([x*x], x, y, domain="QQ")
    ideal.exprs[:] = [3*x*y, 0, 6*x*y]
    divisor = GroebnerBasis([0, 2*x, x], (y, x), "ZZ", "grevlex")
    assert ideal.quotient(divisor).exprs == [y]
    integer_ideal = s.groebner([2*x+1], x, domain="ZZ")
    integer_unit = GroebnerBasis([2], (x,), "ZZ", "lex")
    result = integer_ideal.quotient(integer_unit)
    assert result.exprs == [x+s.Rational(1, 2)] and result.domain == "QQ"
    assert integer_ideal.domain == integer_unit.domain == "ZZ"


def test_quotient_preserves_typed_symbol_and_dummy_owners():
    plain = s.Symbol("quotient_identity")
    assumed = s.Symbol("quotient_identity", real=True)
    dummy = s.Dummy("quotient_identity")
    generators = (dummy, assumed, plain)
    ideal = s.groebner([plain*plain*assumed*dummy], *generators, domain="QQ")
    divisor = s.groebner([plain], *reversed(generators), domain="QQ")
    result = ideal.quotient(divisor)
    assert all(a is b for a, b in zip(result.gens, generators))
    assert result.exprs == [plain*assumed*dummy]
    assert all(any(owner is symbol for owner in result.exprs[0].free_symbols)
               for symbol in generators)
    with pytest.raises(ValueError, match="same typed polynomial ring"):
        s.groebner([plain], plain, domain="QQ").quotient(s.groebner([assumed], assumed, domain="QQ"))


@pytest.mark.parametrize("change", ["left_list", "right_list", "generators", "domain",
                                    "held_children", "native_value", "assumptions"])
def test_quotient_rejects_source_mutation_before_result_publication(monkeypatch, change):
    x, y = s.symbols("quotient_mutation_" + change + "_x quotient_mutation_" + change + "_y")
    held = s.Add(x, y, evaluate=False)
    ideal = GroebnerBasis([held], (x, y), "QQ", "lex")
    divisor = GroebnerBasis([1], (x, y), "QQ", "lex")
    native = _native.ideal_quotient_native

    def mutate(*args):
        result = native(*args)
        if change == "left_list":
            ideal.exprs.append(x)
        elif change == "right_list":
            divisor.exprs.append(x)
        elif change == "generators":
            ideal.gens = (y, x)
        elif change == "domain":
            divisor.domain = "ZZ"
        elif change == "held_children":
            held._args = (x, x)
        elif change == "native_value":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "ideal_quotient_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        ideal.quotient(divisor)


def test_quotient_checks_sources_again_after_lifting(monkeypatch):
    import sympy.polys.polytools as polytools

    x = s.Symbol("quotient_lift_mutation")
    ideal = GroebnerBasis([x*x], (x,), "QQ", "lex")
    divisor = GroebnerBasis([x], (x,), "QQ", "lex")
    lift = polytools._lift_native_result

    def mutate(*args):
        result = lift(*args)
        divisor.exprs[:] = [s.Integer(0)]
        return result

    monkeypatch.setattr(polytools, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        ideal.quotient(divisor)


def test_quotient_rejects_foreign_protocols_before_invoking_them():
    class EffectfulMeta(type):
        def __eq__(cls, other):
            raise AssertionError("foreign class equality must not run")

    class Foreign(metaclass=EffectfulMeta):
        pass

    class CustomBasis(GroebnerBasis):
        def __getattribute__(self, name):
            raise AssertionError("custom basis attributes must not run")

    x = s.Symbol("quotient_closed_source")
    ideal = GroebnerBasis([x], (x,), "QQ", "lex")
    with pytest.raises(ValueError, match="exact rational polynomial expressions"):
        ideal.quotient(GroebnerBasis([Foreign()], (x,), "QQ", "lex"))
    with pytest.raises(TypeError, match="exact GroebnerBasis"):
        ideal.quotient(object.__new__(CustomBasis))


def test_quotient_poly_sources_use_captured_storage_without_callbacks(monkeypatch):
    x, y = s.symbols("quotient_poly_source_x quotient_poly_source_y")
    polynomial = s.Poly(x*y, x, y)
    ideal = GroebnerBasis([polynomial], (x, y), "QQ", "lex")
    divisor = GroebnerBasis([x], (x, y), "QQ", "lex")
    assert ideal.quotient(divisor).exprs == [y]
    calls = []

    def foreign():
        calls.append("as_expr")
        return x

    monkeypatch.setattr(polynomial, "as_expr", foreign)
    with pytest.raises(TypeError, match="overridden Poly.as_expr"):
        ideal.quotient(divisor)
    assert calls == []


@pytest.mark.parametrize("change", ["children", "native_children", "native_operator"])
def test_quotient_rejects_inconsistent_sources_that_predate_the_call(monkeypatch, change):
    x, y, z = s.symbols("quotient_old_mutation_x quotient_old_mutation_y quotient_old_mutation_z")
    source = s.Add(x, y, evaluate=False)
    if change == "children":
        source._args = (x, z)
    elif change == "native_children":
        source._value = _native_expr(s.Add(x, z, evaluate=False))
    else:
        source._value = _native_expr(s.Mul(x, y, evaluate=False))
    ideal = GroebnerBasis([source], (x, y, z), "QQ", "lex")
    divisor = GroebnerBasis([1], (x, y, z), "QQ", "lex")

    def generated_too_early(*args):
        raise AssertionError("inconsistent source reached quotient generation")

    monkeypatch.setattr(_native, "ideal_quotient_native", generated_too_early)
    with pytest.raises(ValueError, match="retained surface arguments"):
        ideal.quotient(divisor)


def test_quotient_accepts_unchanged_held_identity_and_nested_sources():
    x = s.Symbol("quotient_held_identity")
    unit = GroebnerBasis([1], (x,), "QQ", "lex")
    cases = [
        (s.Add(x, 0, evaluate=False), x),
        (s.Add(x, x, evaluate=False), x),
        (s.Mul(1, x, evaluate=False), x),
        (s.Pow(x, 1, evaluate=False), x),
        (s.Pow(x, 0, evaluate=False), s.Integer(1)),
        (s.Mul(s.Add(x, 0, evaluate=False), s.Pow(x, 1, evaluate=False), evaluate=False), x*x),
    ]
    for source, expected in cases:
        assert GroebnerBasis([source], (x,), "QQ", "lex").quotient(unit).exprs == [expected]


def test_saturation_removes_a_whole_component_and_preserves_multiplicity():
    x, y = s.symbols("saturation_x saturation_y")
    assert s.groebner([x*x*y], x, y, domain="QQ").saturate(x).exprs == [y]
    assert s.groebner([x*y*y], x, y, domain="QQ").saturate(x).exprs == [y*y]
    assert s.groebner([x*x-y*y], x, y, domain="QQ").saturate(x-y).exprs == [x+y]
    assert s.groebner([x*(y*y+1)], x, y, domain="QQ").saturate(y*y+1).exprs == [x]


def test_saturation_is_an_ideal_closure_not_a_domain_exclusion_set():
    x, y = s.symbols("closure_x closure_y")
    result = s.groebner([x*y], x, y, domain="QQ").saturate(x)
    assert result.exprs == [y]
    # The origin belongs to the resulting closed variety even though x=0
    # was inverted in the auxiliary ideal. A solver must keep exclusions.
    assert all(expression.subs({x: 0, y: 0}) == 0 for expression in result)


def test_saturation_by_zero_unit_and_scalar_multiples_is_exact():
    x, y = s.symbols("saturation_edge_x saturation_edge_y")
    ideal = s.groebner([x*y], x, y, domain="QQ")
    assert ideal.saturate(0).exprs == [1]
    assert ideal.saturate(1).exprs == ideal.exprs
    assert ideal.saturate(2*x).exprs == ideal.saturate(x).exprs
    assert ideal.saturate(x*y).exprs == [1]


def test_integer_coefficients_are_explicitly_extended_to_QQ():
    x = s.Symbol("field_extension_x")
    basis = s.groebner([2*x+1], x, domain="ZZ")
    result = basis.saturate(2)
    assert basis.domain == "ZZ"
    assert result.domain == "QQ"
    assert result.exprs == [x+s.Rational(1, 2)]


def test_same_printed_names_do_not_alias_typed_ideal_generators():
    plain = s.Symbol("ideal_identity")
    assumed = s.Symbol("ideal_identity", real=True)
    basis = s.groebner([plain-assumed*assumed], plain, assumed, domain="QQ")
    result = basis.eliminate(plain)
    assert result.gens == (assumed,) and result.gens[0] is assumed
    assert result.exprs == []
    with pytest.raises(ValueError):
        s.groebner([plain], plain, domain="QQ").eliminate(assumed)
    with pytest.raises(ValueError):
        s.groebner([plain], plain, domain="QQ").intersection(s.groebner([assumed], assumed, domain="QQ"))


@pytest.mark.parametrize("exponent", [-1, 0])
def test_undefined_original_children_are_admitted_before_unit_or_empty_shortcuts(exponent):
    x = s.Symbol("ideal_held_x")
    undefined = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    malformed = GroebnerBasis([1, undefined], (x,), "QQ", "lex")
    zero = s.groebner([], x, domain="QQ")
    for operation in [lambda: malformed.eliminate(x), lambda: malformed.intersection(zero),
                      lambda: malformed.saturate(0), lambda: malformed.quotient(zero),
                      lambda: zero.quotient(malformed)]:
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError):
        s.groebner([1], x, domain="QQ").saturate(undefined)


def test_approximate_custom_and_foreign_sources_refuse_the_exact_lane():
    class CustomSymbol(s.Symbol):
        pass

    x, y = s.symbols("ideal_bad_x ideal_bad_y")
    ideal = s.groebner([x], x, domain="QQ")
    for source in [s.Float("1.25"), s.sin(x), y, s.Pow(x, -1, evaluate=False)]:
        with pytest.raises(ValueError):
            ideal.saturate(source)
        with pytest.raises(ValueError):
            ideal.quotient(GroebnerBasis([source], (x,), "QQ", "lex"))
    with pytest.raises(NotImplementedError):
        ideal.eliminate(CustomSymbol("ideal_bad_x"))
    with pytest.raises(ValueError):
        ideal.eliminate(x, x)


def test_native_failure_does_not_fall_back_to_unchecked_ideal_operations(monkeypatch):
    x, y = s.symbols("ideal_budget_x ideal_budget_y")
    ideal = s.groebner([x*y], x, y, domain="QQ")

    def exhausted(*args):
        raise ValueError("ideal operation budget exhausted")

    monkeypatch.setattr(_native, "ideal_saturation_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        ideal.saturate(x)
    monkeypatch.setattr(_native, "ideal_quotient_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        ideal.quotient(GroebnerBasis([x], (x, y), "QQ", "lex"))


def test_native_ideal_boundary_checks_lengths_and_typed_elimination():
    x, y = s.symbols("ideal_native_x ideal_native_y")
    source = [_native_expr(x-y)]
    with pytest.raises(ValueError):
        _native.ideal_eliminate_native(source, [_symbol_binding(x), _symbol_binding(y)], [_symbol_binding(x), _symbol_binding(x)])
    with pytest.raises(ValueError):
        _native.ideal_intersection_native(source*65, [], [_symbol_binding(x), _symbol_binding(y)])
    for left, right in [(source*65, []), ([], source*65)]:
        with pytest.raises(ValueError):
            _native.ideal_quotient_native(left, right, [_symbol_binding(x), _symbol_binding(y)])
    with pytest.raises(ValueError):
        _native.ideal_quotient_native([], [], [_symbol_binding(x), _symbol_binding(x)])


def test_raw_integer_bounds_precede_native_integer_conversion(monkeypatch):
    x = s.Symbol("ideal_integer_bound_x")
    ideal = GroebnerBasis([x], (x,), "QQ", "lex")

    def copied_too_early(*args):
        raise AssertionError("oversized integer was copied before its bit bound")

    monkeypatch.setattr(_native, "py_integer", copied_too_early)
    for coefficient in [1 << 4096, -(1 << 4096)]:
        with pytest.raises(ValueError, match="4096 bits"):
            ideal.saturate(coefficient)


def test_exact_atom_bounds_borrow_values_without_exporting_numerators(monkeypatch):
    x = s.Symbol("ideal_atom_bound_x")
    ideal = GroebnerBasis([x], (x,), "QQ", "lex")
    oversized = [s.Integer(1 << 4096), s.Rational(1 << 4096, 3), s.Rational(1, 1 << 4096)]
    largest_admitted = [s.Integer(1 << 4095), s.Rational(1, 1 << 4095)]

    def copied_too_early(self):
        raise AssertionError("bit admission exported an arbitrary-size integer")

    monkeypatch.setattr(s.Rational, "p", property(copied_too_early))
    monkeypatch.setattr(s.Rational, "q", property(copied_too_early))
    monkeypatch.setattr(s.Integer, "p", property(copied_too_early))
    for coefficient in oversized:
        with pytest.raises(ValueError, match="4096 bits"):
            ideal.saturate(coefficient)
    for coefficient in largest_admitted:
        assert _native.ideal_source_preflight_native(_native_expr(coefficient)) is None


def test_native_source_bounds_run_before_python_args_clone_held_children(monkeypatch):
    x = s.Symbol("ideal_child_bound_x")
    held = s.Add(x, s.Integer(1 << 4096), evaluate=False)
    malformed = GroebnerBasis([held], (x,), "QQ", "lex")

    def copied_too_early(self):
        raise AssertionError("source traversal copied children before coefficient admission")

    monkeypatch.setattr(s.Add, "args", property(copied_too_early))
    with pytest.raises(ValueError, match="4096 bits"):
        malformed.eliminate(x)


def test_source_admission_does_not_lift_native_children_and_keeps_held_type_guards(monkeypatch):
    class CustomSymbol(s.Symbol):
        pass

    x, y = s.symbols("ideal_borrow_x ideal_borrow_y")
    source = x+y
    ideal = GroebnerBasis([source], (x, y), "QQ", "lex")
    held = s.Add(x, 0, evaluate=False)
    held._args = (CustomSymbol("ideal_borrow_x"), s.Integer(0))

    def copied_too_early(self):
        raise AssertionError("native-owned children must remain borrowed during admission")

    monkeypatch.setattr(s.Add, "args", property(copied_too_early))
    assert ideal.eliminate(x).exprs == []
    with pytest.raises(ValueError, match="exact rational polynomial expressions"):
        GroebnerBasis([held], (x,), "QQ", "lex").eliminate(x)


def test_native_ring_name_bounds_cover_all_symbol_forms_and_utf8_bytes():
    class EffectfulName(str):
        def __len__(self):
            raise AssertionError("custom name methods must not run during admission")

    with pytest.raises(ValueError):
        _native.ideal_eliminate_native([], [EffectfulName("x")], [])
    for name in ["x" * 65_537, "λ" * 32_769]:
        symbol = _native.Symbol(name)
        for variable in [name, _native.SymbolBinding(name), symbol, symbol.as_expr()]:
            with pytest.raises(ValueError, match="65536 bytes"):
                _native.ideal_eliminate_native([], [variable], [])
    for variables in [
        ["x" * 40_000, "y" * 40_000],
        [_native.SymbolBinding("x" * 40_000), _native.SymbolBinding("y" * 40_000)],
    ]:
        with pytest.raises(ValueError, match="65536 bytes"):
            _native.ideal_eliminate_native([], variables, [])
