"""Complete relations over prime fields with ordered original coordinates.

Known complete modules were independently checked with SymPy 1.14.0 AGCA over
GF(2), GF(3), and GF(5). Comparators check module spans, allowing the documented
nonminimal generators and avoiding rational-rank tests in finite characteristic.
"""

from itertools import combinations

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys.polyerrors import CoercionFailed, DomainError, UnificationFailed


ORDERS = ("lex", "grlex", "grevlex")
BRIDGE = "prime_field_polynomial_syzygies_native"


def _basis(sources, gens, prime=5, order="lex", symmetric=True, polys=False):
    result = sp.groebner([], *gens, modulus=prime, order=order,
                        symmetric=symmetric, polys=polys)
    result.exprs[:] = sources
    return result


def _zero(expression, gens, prime):
    return (sp.Poly(expression, *gens, modulus=prime).is_zero if gens
            else int(expression) % prime == 0)


def _relations(basis, rows):
    assert type(rows) is list
    for row in rows:
        assert type(row) is tuple and len(row) == len(basis.exprs)
        assert any(value != 0 for value in row)
        expression = sum((value*source for value, source in zip(row, basis.exprs)), sp.Integer(0))
        assert _zero(expression, basis.gens, basis._modulus)


def _divide(expression, divisor, gens, prime):
    quotient, remainder = sp.Poly(expression, *gens, modulus=prime).div(
        sp.Poly(divisor, *gens, modulus=prime))
    assert remainder.is_zero
    return quotient.as_expr()


@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("common_factor", [False, True])
def test_two_inputs_span_the_complete_known_principal_module(prime, order, common_factor):
    x, y = sp.symbols("field_relation_pair_x field_relation_pair_y")
    sources = [x*y, x*(y+1)] if common_factor else [x, y]
    primitive = (y+1, -y) if common_factor else (y, -x)
    basis = _basis(sources, (x, y), prime, order)
    rows = basis.syzygies()
    _relations(basis, rows)
    multipliers = []
    for a, b in rows:
        q = _divide(a, primitive[0], basis.gens, prime)
        assert _zero(b-q*primitive[1], basis.gens, prime)
        multipliers.append(q)
    # A unit coefficient ideal proves the rows generate every multiple of
    # the primitive relation, not merely some true polynomial identities.
    assert sp.groebner(multipliers, *basis.gens, modulus=prime).contains(1)


@pytest.mark.parametrize("prime", [2, 5])
@pytest.mark.parametrize("order", ORDERS)
def test_three_monomials_span_the_complete_rank_two_relation_module(prime, order):
    x, y = sp.symbols("field_relation_three_x field_relation_three_y")
    basis = _basis([x*x, x*y, y*y], (x, y), prime, order)
    rows = basis.syzygies()
    _relations(basis, rows)
    coordinates = []
    for a, b, c in rows:
        u = _divide(a, y, basis.gens, prime)
        v = _divide(-c, x, basis.gens, prime)
        assert _zero(b+u*x-v*y, basis.gens, prime)
        coordinates.append((u, v))
    minors = [a*d-b*c for (a, b), (c, d) in combinations(coordinates, 2)]
    # Surjectivity in the known free basis (y,-x,0), (0,y,-x) is equivalent
    # to the maximal minors generating the unit ideal over this field.
    assert sp.groebner(minors, *basis.gens, modulus=prime).contains(1)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_current_zero_duplicate_and_scaled_positions_are_not_normalized_away(prime):
    x = sp.Symbol("field_relation_coordinates_x")
    basis = _basis([0, x, 0, 2*x], (x,), prime)
    original = tuple(basis.exprs)
    rows = basis.syzygies()
    _relations(basis, rows)
    assert tuple(basis.exprs) == original
    assert all(not value.free_symbols for row in rows for value in row)
    # Coordinates0,2,3 freely specify every relation, including when p=2
    # turns the fourth input into an additional zero generator.
    coordinates = [(int(row[0]), int(row[2]), int(row[3])) for row in rows]
    assert any(int(sp.Matrix(minor).det()) % prime for minor in combinations(coordinates, 3))


@pytest.mark.parametrize("prime,expected", [(2, (1, 0)), (3, (0, 1))])
def test_characteristic_changes_which_original_coordinate_is_free(prime, expected):
    x, y = sp.symbols("field_relation_characteristic_x field_relation_characteristic_y")
    basis = _basis([2*x, 3*y], (x, y), prime)
    rows = basis.syzygies()
    _relations(basis, rows)
    assert rows == [expected]


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_relation_coefficients_keep_residue_policy_and_expression_tuple_output(polys, symmetric):
    x, y = sp.symbols("field_relation_policy_x field_relation_policy_y")
    basis = _basis([x, y], (x, y), symmetric=symmetric, polys=polys)
    rows = basis.syzygies()
    assert rows == [(y, (-1 if symmetric else 4)*x)]
    assert all(type(row) is tuple and all(not isinstance(value, sp.Poly) for value in row)
               for row in rows)
    assert basis.domain == "GF(5)" and basis._polys is polys
    assert basis._symmetric is symmetric and basis.gens == (x, y)


def test_empty_zero_unit_and_constant_relation_modules_are_complete():
    x = sp.Symbol("field_relation_constant_x")
    assert _basis([], (x,)).syzygies() == []
    assert _basis([1], (x,)).syzygies() == []
    assert _basis([0, 0], (x,)).syzygies() == [(1, 0), (0, 1)]
    unit = _basis([x, 1], (x,))
    assert unit.syzygies() == [(1, -x)]
    constant = sp.groebner([x], x, modulus=5).eliminate(x)
    constant.exprs[:] = [0, 2, 3]
    rows = constant.syzygies()
    _relations(constant, rows)
    coordinates = [(int(row[0]), int(row[2])) for row in rows]
    assert any((a*d-b*c) % 5 for (a, b), (c, d) in combinations(coordinates, 2))


def test_all_32_zero_source_coordinates_survive_the_input_boundary():
    x = sp.Symbol("field_relation_limit_x")
    basis = _basis([0]*32, (x,))
    rows = basis.syzygies()
    assert rows == [tuple(sp.Integer(i == j) for j in range(32)) for i in range(32)]


def test_same_printed_typed_symbols_keep_original_owners_in_every_relation():
    plain = sp.Symbol("field_relation_typed")
    assumed = sp.Symbol("field_relation_typed", positive=True)
    dummy = sp.Dummy("field_relation_typed")
    generators = (dummy, assumed, plain)
    basis = _basis([plain, assumed, dummy], generators)
    rows = basis.syzygies()
    _relations(basis, rows)
    assert rows
    assert all(any(symbol is original for original in generators)
               for row in rows for value in row for symbol in value.free_symbols)


def test_original_poly_sources_keep_declared_rings_and_fields(monkeypatch):
    x, y = sp.symbols("field_relation_poly_x field_relation_poly_y")
    polynomial = sp.Poly(x, x, modulus=5)
    basis = _basis([polynomial, y], (x, y))
    assert basis.syzygies() == [(y, -x)]
    basis.exprs[:] = [sp.Poly(x, x, modulus=7), y]
    with pytest.raises(UnificationFailed):
        basis.syzygies()
    forged = sp.Poly(0, x, modulus=5)
    forged._expr = sp.Add(y, -y, evaluate=False)
    basis.exprs[:] = [forged, 1]
    with pytest.raises(ValueError):
        basis.syzygies()
    calls = []

    def foreign(*args, **kwargs):
        calls.append(1)
        return 0

    monkeypatch.setattr(polynomial, "as_expr", foreign)
    basis.exprs[:] = [polynomial, y]
    with pytest.raises(TypeError, match="Poly.as_expr"):
        basis.syzygies()
    assert calls == []


@pytest.mark.parametrize("shape", ["zero_product", "zero_power"])
def test_original_invalid_children_survive_zero_or_unit_shortcuts(shape):
    x = sp.Symbol("field_relation_hidden_x")
    reciprocal = sp.Pow(x, -1, evaluate=False)
    hidden = (sp.Mul(0, reciprocal, evaluate=False) if shape == "zero_product"
              else sp.Pow(reciprocal, 0, evaluate=False))
    for sources in ([0, hidden], [1, hidden]):
        with pytest.raises(ValueError):
            _basis(sources, (x,)).syzygies()


@pytest.mark.parametrize("source", [sp.Rational(1, 2), sp.Float("1.0")])
def test_nonintegral_and_approximate_sources_refuse_before_relation_generation(source):
    x = sp.Symbol("field_relation_exact_x")
    basis = _basis([1, source], (x,))
    with pytest.raises((CoercionFailed, ValueError)):
        basis.syzygies()


@pytest.mark.parametrize("change", ["expressions", "gens", "order", "domain", "modulus",
                                    "symmetric", "polys", "held_children", "assumptions"])
def test_original_sources_and_controls_are_retained_through_native_work(monkeypatch, change):
    x, y = sp.symbols("field_relation_mutation_x_"+change+" field_relation_mutation_y_"+change)
    held = sp.Add(x, 0, evaluate=False)
    basis = _basis([held, y], (x, y))
    original = getattr(_native, BRIDGE)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "expressions":
            basis.exprs.reverse()
        elif change == "gens":
            basis.gens = (y, x)
        elif change == "order":
            basis.order = "grlex"
        elif change == "domain":
            basis.domain = "QQ"
        elif change == "modulus":
            basis._modulus = 7
        elif change == "symmetric":
            basis._symmetric = False
        elif change == "polys":
            basis._polys = True
        elif change == "held_children":
            held._args = (x, x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, BRIDGE, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.syzygies()


def test_empty_relation_result_is_not_published_after_input_mutation(monkeypatch):
    x = sp.Symbol("field_relation_empty_mutation_x")
    basis = _basis([x], (x,))
    original = getattr(_native, BRIDGE)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result == []
        basis.exprs.append(x)
        return result

    monkeypatch.setattr(_native, BRIDGE, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.syzygies()


def test_midrow_lifting_mutation_prevents_publishing_partial_relations(monkeypatch):
    from sympy.polys import _finitefield_groebner

    x, y = sp.symbols("field_relation_lift_x field_relation_lift_y")
    basis = _basis([x, y], (x, y))
    original = _finitefield_groebner._lift_native_result
    calls = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        basis.exprs.reverse()
        return result

    monkeypatch.setattr(_finitefield_groebner, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.syzygies()
    assert calls == [1]


@pytest.mark.parametrize("failure", [ValueError("original work budget exhausted"), MemoryError("original output budget exhausted")])
def test_native_failure_is_terminal_without_fallback_or_budget_restart(monkeypatch, failure):
    x, y = sp.symbols("field_relation_failure_x field_relation_failure_y")
    basis = _basis([x, y], (x, y))
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure

    def forbidden(*args, **kwargs):
        raise AssertionError("prime-field relations restarted generation or entered QQ")

    monkeypatch.setattr(_native, BRIDGE, refuse)
    monkeypatch.setattr(_native, "polynomial_syzygies_native", forbidden)
    monkeypatch.setattr(_native, "prime_field_groebner_native", forbidden)
    with pytest.raises(type(failure)) as raised:
        basis.syzygies()
    assert raised.value is failure and calls == [1]


def test_inconsistent_field_and_oversized_inputs_refuse_before_mathematics(monkeypatch):
    x = sp.Symbol("field_relation_domain_x")
    basis = _basis([x], (x,))

    def forbidden(*args, **kwargs):
        raise AssertionError("inadmissible source reached relation generation")

    monkeypatch.setattr(_native, BRIDGE, forbidden)
    basis.exprs[:] = [0]*33
    with pytest.raises(ValueError):
        basis.syzygies()
    basis.exprs[:] = [x]
    basis.domain = "QQ"
    with pytest.raises(DomainError):
        basis.syzygies()


@pytest.mark.parametrize("shape", ["wrong_width", "oversized", "empty_coordinate_row"])
def test_result_dimensions_are_checked_before_lifting_any_cell(monkeypatch, shape):
    from sympy.polys import _finitefield_groebner

    x = sp.Symbol("field_relation_shape_x")
    basis = _basis([] if shape == "empty_coordinate_row" else [0]*32, (x,))
    zero = _native_expr(0)
    raw = ([[]] if shape == "empty_coordinate_row" else [[zero]*31] if shape == "wrong_width"
           else [[zero]*32]*513)

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed complete relation shape reached lifting")

    monkeypatch.setattr(_native, BRIDGE, lambda *args, **kwargs: raw)
    monkeypatch.setattr(_finitefield_groebner, "_lift_native_result", forbidden)
    with pytest.raises(ValueError, match="dimension"):
        basis.syzygies()


def test_raw_bridge_admits_empty_rings_and_every_original_source_before_shortcuts():
    x = sp.Symbol("field_relation_raw_x")
    bridge = getattr(_native, BRIDGE)
    bindings = [_symbol_binding(x)]
    assert bridge([], bindings, 5) == []
    for sources, variables, prime in [
        ([], bindings, 9),
        ([], bindings*2, 5),
        ([_native_expr(0)]*33, bindings, 5),
        ([_native_expr(1), _native_expr(sp.sin(x))], bindings, 5),
    ]:
        with pytest.raises(ValueError):
            bridge(sources, variables, prime)
    with pytest.raises(ValueError):
        bridge([], bindings, 5, source_variables=[None])
