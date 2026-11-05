"""Complete polynomial relations and inverses in exact rational quotient rings."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys.polyerrors import NotInvertible, PolynomialError
from sympy.polys.polytools import GroebnerBasis


def assert_relations(inputs, rows):
    assert type(rows) is list
    for row in rows:
        assert type(row) is tuple and len(row) == len(inputs)
        assert any(value != 0 for value in row)
        assert s.expand(sum(value * source for value, source in zip(row, inputs))) == 0


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_two_coordinate_relations_generate_the_full_principal_module(order):
    x, y = s.symbols("relation_pair_x relation_pair_y")
    basis = GroebnerBasis([x, y], (x, y), "QQ", order)
    rows = basis.syzygies()
    assert_relations(basis.exprs, rows)
    # Every relation is q*(y,-x); unit ideal of the q's proves that the
    # returned family spans this whole known module, not just some relations.
    multipliers = []
    for a, b in rows:
        q, remainder = s.div(a, y, x, y)
        assert remainder == 0 and s.expand(b + q*x) == 0
        multipliers.append(q)
    assert s.groebner(multipliers, x, y, domain="QQ").contains(1)


def test_three_monomials_have_complete_rank_two_relations():
    x, y = s.symbols("relation_three_x relation_three_y")
    basis = GroebnerBasis([x*x, x*y, y*y], (x, y), "QQ", "grevlex")
    rows = basis.syzygies()
    assert_relations(basis.exprs, rows)
    # The known free generators are (y,-x,0) and (0,y,-x). Express every
    # returned row in these coordinates and require their maximal minors to
    # generate the unit ideal, proving that they span both generators.
    coordinates = []
    for a, b, c in rows:
        u, remainder_a = s.div(a, y, x, y)
        v, remainder_c = s.div(-c, x, x, y)
        assert remainder_a == remainder_c == 0
        assert s.expand(b + u*x - v*y) == 0
        coordinates.append((u, v))
    minors = [s.expand(a*d-b*c)
              for i, (a, b) in enumerate(coordinates)
              for c, d in coordinates[i+1:]]
    assert s.groebner(minors, x, y, domain="QQ").contains(1)


def test_relations_preserve_current_duplicate_and_zero_input_coordinates():
    x = s.Symbol("relation_current_x")
    basis = s.groebner([x], x, domain="QQ")
    basis.exprs[:] = [x, 0, x]
    rows = basis.syzygies()
    assert_relations(basis.exprs, rows)
    assert all(value.free_symbols == set() for row in rows for value in row)
    assert s.Matrix(rows).rank() == 2
    assert basis.exprs == [x, 0, x]


def test_empty_zero_and_constant_relation_modules():
    x = s.Symbol("relation_edge_x")
    assert GroebnerBasis([], (x,), "QQ", "lex").syzygies() == []
    zeros = GroebnerBasis([0, 0], (x,), "QQ", "lex").syzygies()
    assert set(zeros) == {(s.Integer(1), s.Integer(0)), (s.Integer(0), s.Integer(1))}
    assert GroebnerBasis([3], (), "QQ", "lex").syzygies() == []
    constants = GroebnerBasis([s.Rational(2, 3), 5, 0], (), "QQ", "lex")
    rows = constants.syzygies()
    assert_relations(constants.exprs, rows)
    assert s.Matrix(rows).rank() == 2


def test_non_groebner_inputs_and_rational_field_extension():
    x, y = s.symbols("relation_rational_x relation_rational_y")
    basis = GroebnerBasis([2*(x+y), 3*(x+y), 5*x*y], (x, y), "ZZ", "grlex")
    rows = basis.syzygies()
    assert rows
    assert_relations(basis.exprs, rows)
    assert basis.domain == "ZZ" and basis.order == "grlex"


def test_inverse_supports_positive_dimensional_multivariate_quotients():
    x, y, z = s.symbols("inverse_positive_x inverse_positive_y inverse_positive_z")
    basis = s.groebner([x*y-1], x, y, z, domain="QQ")
    assert not basis.is_zero_dimensional
    assert basis.invert(x) == y
    assert basis.invert(y) == x
    assert basis.reduce(x*basis.invert(x)-1)[1] == 0


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_inverse_is_reduced_in_the_declared_order(order):
    x, y = s.symbols("inverse_order_x inverse_order_y")
    basis = s.groebner([x*x-2, y-x], x, y, domain="QQ", order=order)
    inverse = basis.invert(x+y)
    assert basis.reduce((x+y)*inverse-1)[1] == 0
    assert basis.reduce(inverse)[1] == inverse
    assert inverse == y/4


def test_inverse_handles_nilpotents_and_nonradical_ideals():
    x, y = s.symbols("inverse_nilpotent_x inverse_nilpotent_y")
    basis = s.groebner([x*x, y*y], x, y, domain="QQ")
    assert basis.invert(1-x-y) == 1+x+y+2*x*y
    assert basis.reduce((1-x-y)*basis.invert(1-x-y)-1)[1] == 0


def test_certified_nonunits_raise_the_public_polynomial_exception():
    x, y = s.symbols("inverse_nonunit_x inverse_nonunit_y")
    assert issubclass(NotInvertible, PolynomialError)
    for basis, element in [
        (GroebnerBasis([], (x,), "QQ", "lex"), x),
        (s.groebner([x*x], x, domain="QQ"), x),
        (s.groebner([x*y], x, y, domain="QQ"), x+y),
        (s.groebner([x*x+1], x, domain="QQ"), 0),
    ]:
        with pytest.raises(NotInvertible, match="not invertible"):
            basis.invert(element)


def test_inverse_unit_zero_and_constant_ring_conventions():
    x = s.Symbol("inverse_edge_x")
    unit = GroebnerBasis([1], (x,), "QQ", "lex")
    assert unit.invert(0) == unit.invert(x) == 0
    zero = GroebnerBasis([], (x,), "QQ", "lex")
    assert zero.invert(s.Rational(2, 3)) == s.Rational(3, 2)
    constants = GroebnerBasis([], (), "QQ", "lex")
    assert constants.invert(s.Rational(-3, 2)) == s.Rational(-2, 3)
    with pytest.raises(NotInvertible):
        constants.invert(0)
    assert GroebnerBasis([2], (), "ZZ", "lex").invert(0) == 0


def test_inverse_admits_current_generators_and_extends_ZZ_to_QQ():
    x = s.Symbol("inverse_current_x")
    basis = s.groebner([x*x+1], x, domain="QQ")
    assert basis.invert(x) == -x
    basis.exprs[:] = [0, 2*x+1, 4*x+2]
    assert basis.invert(x) == -2
    integer_basis = GroebnerBasis([2*x+1], (x,), "ZZ", "grevlex")
    assert integer_basis.invert(2*x) == -1
    assert integer_basis.domain == "ZZ"


def test_results_preserve_typed_symbols_and_dummy_owners():
    plain = s.Symbol("relation_identity")
    assumed = s.Symbol("relation_identity", real=True)
    dummy = s.Dummy("relation_identity")
    gens = (dummy, assumed, plain)
    basis = GroebnerBasis([plain, assumed, dummy], gens, "QQ", "lex")
    rows = basis.syzygies()
    assert_relations(basis.exprs, rows)
    for row in rows:
        for value in row:
            assert all(any(symbol is original for original in gens) for symbol in value.free_symbols)
    inverse_basis = GroebnerBasis([plain*assumed-1, plain-dummy], gens, "QQ", "lex")
    inverse = inverse_basis.invert(plain)
    assert inverse == assumed
    assert next(iter(inverse.free_symbols)) is assumed


@pytest.mark.parametrize("operation", ["syzygies", "invert"])
def test_original_undefined_sources_refuse_even_in_zero_or_unit_shortcuts(operation):
    x = s.Symbol("relation_undefined_x")
    undefined = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    malformed = GroebnerBasis([1, undefined], (x,), "QQ", "lex")
    with pytest.raises(ValueError):
        malformed.syzygies() if operation == "syzygies" else malformed.invert(0)
    if operation == "invert":
        with pytest.raises(ValueError):
            GroebnerBasis([1], (x,), "QQ", "lex").invert(undefined)


def test_unsupported_sources_and_nonliteral_protocols_do_not_enter_checked_generation():
    class CustomSymbol(s.Symbol):
        pass

    x, foreign = s.symbols("relation_admit_x relation_admit_foreign")
    basis = GroebnerBasis([x], (x,), "QQ", "lex")
    for expression in [s.Float("1.25"), s.sin(x), foreign, s.Pow(x, -1, evaluate=False),
                       CustomSymbol("relation_custom")]:
        with pytest.raises((ValueError, NotImplementedError)):
            GroebnerBasis([expression], (x,), "QQ", "lex").syzygies()
        with pytest.raises((ValueError, NotImplementedError)):
            basis.invert(expression)
    with pytest.raises(NotImplementedError):
        GroebnerBasis([x], (x,), "RR", "lex").syzygies()
    with pytest.raises(ValueError):
        GroebnerBasis([x], (x,), "QQ", "unknown").invert(1)


@pytest.mark.parametrize("operation,native_name", [
    ("syzygies", "polynomial_syzygies_native"),
    ("invert", "ideal_inverse_native"),
])
def test_source_mutation_before_publication_refuses_the_result(monkeypatch, operation, native_name):
    x, y = s.symbols("relation_mutation_x relation_mutation_y")
    basis = GroebnerBasis([x*y-1], (x, y), "QQ", "lex")
    native = getattr(_native, native_name)

    def mutate(*args):
        result = native(*args)
        basis.exprs.append(x)
        return result

    monkeypatch.setattr(_native, native_name, mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        basis.syzygies() if operation == "syzygies" else basis.invert(x)


@pytest.mark.parametrize("operation", ["syzygies", "invert"])
def test_source_mutation_during_lifting_refuses_the_result(monkeypatch, operation):
    from sympy.polys import polytools

    x, y = s.symbols("relation_lift_x relation_lift_y")
    basis = GroebnerBasis([x, y], (x, y), "QQ", "lex")
    lift = polytools._lift_native_result

    def mutate(*args):
        result = lift(*args)
        basis.order = "grlex"
        return result

    monkeypatch.setattr(polytools, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        basis.syzygies() if operation == "syzygies" else basis.invert(1)


def test_inverse_poly_element_uses_retained_source_without_callbacks(monkeypatch):
    x = s.Symbol("inverse_poly_source_x")
    basis = GroebnerBasis([x*x+1], (x,), "QQ", "lex")
    polynomial = s.Poly(x, x)
    assert basis.invert(polynomial) == -x
    calls = []

    def foreign(*args):
        calls.append(True)
        return 1

    monkeypatch.setattr(polynomial, "as_expr", foreign)
    with pytest.raises(TypeError, match="Poly.as_expr"):
        basis.invert(polynomial)
    assert calls == []


@pytest.mark.parametrize("operation,native_name", [
    ("syzygies", "polynomial_syzygies_native"),
    ("invert", "ideal_inverse_native"),
])
def test_inconsistent_held_source_is_rejected_before_native_generation(monkeypatch, operation, native_name):
    x, y = s.symbols("relation_held_x relation_held_y")
    source = s.Add(x, y, evaluate=False)
    source._args = (x, x)
    basis = GroebnerBasis([source], (x, y), "QQ", "lex")

    def too_early(*args):
        raise AssertionError("an inconsistent retained source reached checked generation")

    monkeypatch.setattr(_native, native_name, too_early)
    with pytest.raises(ValueError, match="retained surface arguments"):
        basis.syzygies() if operation == "syzygies" else basis.invert(1)


@pytest.mark.parametrize("operation,native_name", [
    ("syzygies", "polynomial_syzygies_native"),
    ("invert", "ideal_inverse_native"),
])
def test_operational_errors_remain_terminal_and_are_not_nonunit_claims(monkeypatch, operation, native_name):
    x = s.Symbol("relation_budget_x")
    basis = GroebnerBasis([x*x+1], (x,), "QQ", "lex")

    def exhausted(*args):
        raise ValueError("polynomial relation work exhausted")

    monkeypatch.setattr(_native, native_name, exhausted)
    with pytest.raises(ValueError, match="work exhausted"):
        basis.syzygies() if operation == "syzygies" else basis.invert(x)


def test_native_boundary_checks_input_limits_and_original_ring_before_shortcuts():
    x = s.Symbol("relation_native_x")
    source = [_native_expr(x)]
    bindings = [_symbol_binding(x)]
    with pytest.raises(ValueError):
        _native.polynomial_syzygies_native(source*65, bindings, "lex")
    with pytest.raises(ValueError):
        _native.ideal_inverse_native(source*64, _native_expr(1), bindings, "lex")
    with pytest.raises(ValueError):
        _native.polynomial_syzygies_native([], bindings*2, "lex")
    with pytest.raises(ValueError):
        _native.ideal_inverse_native([], _native_expr(1), bindings*2, "lex")
    with pytest.raises(ValueError):
        _native.polynomial_syzygies_native(source, bindings, "invalid")
