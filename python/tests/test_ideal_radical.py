"""Complete ideal radicals, with independently derived algebraic expectations."""

import pytest

import sympy as s
from sympy.core import _native, _native_expr
from sympy.polys.polytools import GroebnerBasis


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_nonreduced_point_ideal_loses_all_nilpotent_directions(order):
    x, y = s.symbols("radical_point_x radical_point_y")
    original = GroebnerBasis([x**2, x*y, y**3], (x, y), "QQ", order)
    result = original.radical()
    assert type(result) is GroebnerBasis
    assert result.exprs == [x, y]
    assert result.gens == (x, y) and result.domain == "QQ" and result.order == "lex"
    assert original.exprs == [x**2, x*y, y**3]
    assert original.is_radical is False and result.is_radical is True
    assert result.radical() == result
    assert not original.contains(x) and not original.contains(y)


@pytest.mark.parametrize("multiplicity", [2, 3, 5])
def test_coupled_irrational_points_are_preserved_without_multiplicity(multiplicity):
    x, y = s.symbols("radical_coupled_x radical_coupled_y")
    original = GroebnerBasis([(x**2-2)**multiplicity, y-x], (x, y), "QQ", "grevlex")
    result = original.radical()
    assert result.exprs == [x-y, y**2-2]
    assert not result.contains(x) and not result.contains(y)
    assert result.contains(x**2-2) and result.contains(y-x)
    assert original.is_radical is False and result.is_radical is True


def test_cartesian_product_keeps_all_four_points():
    x, y = s.symbols("radical_product_x radical_product_y")
    original = GroebnerBasis([(x*x-1)**2, (y*y-1)**3], (x, y), "QQ", "lex")
    result = original.radical()
    assert result.exprs == [x*x-1, y*y-1]
    assert len(result.quotient_basis()) == 4
    assert not result.contains(x-y) and not result.contains(x+y)
    for a in (-1, 1):
        for b in (-1, 1):
            assert all(value.subs({x: a, y: b}) == 0 for value in result.exprs)


def test_complex_points_do_not_become_an_empty_real_solution_ideal():
    x = s.Symbol("radical_complex_x")
    original = GroebnerBasis([(x*x+1)**2], (x,), "QQ", "lex")
    result = original.radical()
    assert result.exprs == [x*x+1]
    assert not result.contains(1) and result.is_radical is True
    assert original.is_radical is False


def test_integer_coefficients_extend_to_QQ_before_radical_computation():
    x = s.Symbol("radical_integer_x")
    original = GroebnerBasis([4*x*x+4*x+1], (x,), "ZZ", "grlex")
    result = original.radical()
    assert result.exprs == [x+s.Rational(1, 2)]
    assert result.domain == "QQ" and original.domain == "ZZ"
    assert original.is_radical is False


def test_current_input_list_is_the_original_ideal_even_when_not_a_basis():
    x, y = s.symbols("radical_current_x radical_current_y")
    original = s.groebner([x], x, y, domain="QQ")
    original.exprs[:] = [0, 2*x*x, x*y, 3*y*y, x*x]
    result = original.radical()
    assert result.exprs == [x, y]
    assert original.exprs == [0, 2*x*x, x*y, 3*y*y, x*x]


def test_typed_generators_and_dummy_owners_survive_publication():
    plain = s.Symbol("radical_typed")
    real = s.Symbol("radical_typed", real=True)
    dummy = s.Dummy("radical_typed")
    generators = (dummy, real, plain)
    original = GroebnerBasis([plain**2, real**3, dummy**2], generators, "QQ", "lex")
    result = original.radical()
    assert all(a is b for a, b in zip(result.gens, generators))
    assert all(a is b for a, b in zip(result.exprs, generators))
    assert original.is_radical is False and result.is_radical is True


@pytest.mark.parametrize("sources", [[1], [0, 2, 0]])
def test_unit_ideal_is_radical_and_admitted_in_its_original_ring(sources):
    x, y = s.symbols("radical_unit_x radical_unit_y")
    original = GroebnerBasis(sources, (x, y), "QQ", "lex")
    assert original.radical().exprs == [1]
    assert original.is_radical is True


@pytest.mark.parametrize("sources", [[], [0], ["x"], ["xy"]])
def test_positive_dimensional_ideals_raise_instead_of_false_or_partial_results(sources):
    x, y = s.symbols("radical_dimension_x radical_dimension_y")
    sources = [x if value == "x" else x*y if value == "xy" else value for value in sources]
    original = GroebnerBasis(sources, (x, y), "QQ", "lex")
    with pytest.raises(ValueError, match="zero-dimensional"):
        original.radical()
    with pytest.raises(ValueError, match="zero-dimensional"):
        _ = original.is_radical


@pytest.mark.parametrize("kind", ["pole", "held_cancel", "nonfinite", "undeclared"])
def test_unit_generator_cannot_hide_an_invalid_original_source(kind):
    x, y = s.symbols("radical_hidden_x radical_hidden_y")
    if kind == "pole":
        bad = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    elif kind == "held_cancel":
        bad = s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False)
    elif kind == "nonfinite":
        bad = s.oo
    else:
        bad = y
    original = GroebnerBasis([1, bad], (x,), "QQ", "lex")
    with pytest.raises((ValueError, NotImplementedError)):
        original.radical()
    with pytest.raises((ValueError, NotImplementedError)):
        _ = original.is_radical


@pytest.mark.parametrize("change", ["list", "generators", "domain", "order", "held", "native", "assumptions"])
def test_source_mutation_during_native_work_prevents_publication(monkeypatch, change):
    x = s.Symbol("radical_mutation_"+change)
    held = s.Pow(x, 2, evaluate=False)
    original = GroebnerBasis([held], (x,), "QQ", "lex")
    native = _native.ideal_radical_native

    def mutate(*args):
        result = native(*args)
        if change == "list":
            original.exprs.append(s.Integer(1))
        elif change == "generators":
            original.gens = ()
        elif change == "domain":
            original.domain = "ZZ"
        elif change == "order":
            original.order = "grlex"
        elif change == "held":
            held._args = (x, s.Integer(3))
        elif change == "native":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "ideal_radical_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        original.radical()


def test_final_result_lifting_is_inside_the_source_snapshot(monkeypatch):
    import sympy.polys.polytools as polytools

    x = s.Symbol("radical_lift_mutation")
    original = GroebnerBasis([x*x], (x,), "QQ", "lex")
    lift = polytools._lift_native_result

    def mutate(*args):
        result = lift(*args)
        original.exprs[:] = [s.Integer(1)]
        return result

    monkeypatch.setattr(polytools, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        original.radical()


def test_operational_failure_does_not_become_a_mathematical_answer(monkeypatch):
    x = s.Symbol("radical_operation_failure")
    original = GroebnerBasis([x*x], (x,), "QQ", "lex")

    def exhausted(*args):
        raise ValueError("test resource exhausted")

    monkeypatch.setattr(_native, "ideal_radical_native", exhausted)
    with pytest.raises(ValueError, match="test resource exhausted"):
        original.radical()
    with pytest.raises(ValueError, match="test resource exhausted"):
        _ = original.is_radical


def test_count_height_and_ring_bounds_apply_before_generation():
    x = s.Symbol("radical_limit_x")
    for source in (GroebnerBasis([x]*65, (x,), "QQ", "lex"),
                   GroebnerBasis([s.Integer(1 << 4096)], (x,), "QQ", "lex"),
                   GroebnerBasis([x], (x, x), "QQ", "lex")):
        with pytest.raises(ValueError):
            source.radical()


def test_custom_basis_classes_are_not_silently_lowered():
    class CustomBasis(GroebnerBasis):
        pass

    x = s.Symbol("radical_subclass_x")
    with pytest.raises(TypeError, match="exact GroebnerBasis"):
        CustomBasis([x*x], (x,), "QQ", "lex").radical()
