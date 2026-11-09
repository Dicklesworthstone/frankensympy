"""Public finite-quotient operators and genuine modular FGLM conversion.

The complete basis fixtures are independently checked against SymPy 1.14.0.
Operator fixtures use exact hand-derived quotient identities, including
nilpotents and characteristics for which a rational reinterpretation is false.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr
from sympy.polys.polyerrors import CoercionFailed, DomainError, UnificationFailed
from sympy.polys.polytools import GroebnerBasis


ORDERS = ("lex", "grlex", "grevlex")


def _residues(matrix, prime):
    return sp.Matrix(matrix.rows, matrix.cols, [int(value) % prime for value in matrix])


def _operation(basis, name, element):
    if name == "fglm":
        return basis.fglm("grlex")
    if name == "quotient_basis":
        return basis.quotient_basis()
    return basis.multiplication_matrix(element)


def _bridge(name):
    return ("prime_field_groebner_fglm_native" if name == "fglm"
            else "prime_field_quotient_multiplication_native")


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
@pytest.mark.parametrize("source_order", ORDERS)
@pytest.mark.parametrize("target_order", ORDERS)
def test_fglm_preserves_complete_ideal_across_all_orders(prime, source_order, target_order):
    x, y = sp.symbols("field_fglm_x field_fglm_y")
    basis = sp.groebner([x*y - 1, y*y - x], x, y, modulus=prime, order=source_order)
    original = tuple(basis.exprs)
    converted = basis.fglm(target_order)
    sign = 1 if prime == 2 else -1
    expected = ([x + sign*y*y, y**3 + sign] if target_order == "lex" else
                [x*x + sign*y, x*y + sign, y*y + sign*x])
    assert type(converted) is GroebnerBasis
    assert converted.exprs == expected
    assert converted.gens == (x, y)
    assert converted.gens[0] is x and converted.gens[1] is y
    assert converted.domain == "GF(%d)" % prime and converted.order == target_order
    assert tuple(basis.exprs) == original
    assert (converted is basis) is (source_order == target_order)


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_fglm_retains_field_lift_and_polynomial_output_policies(polys, symmetric):
    x, y = sp.symbols("field_fglm_policy_x field_fglm_policy_y")
    basis = sp.groebner([2*x + y, y*y + 1], x, y, modulus=5,
                        order="grevlex", polys=polys, symmetric=symmetric)
    converted = basis.fglm("lex")
    assert converted.exprs == [x + (-2 if symmetric else 3)*y, y*y + 1]
    assert converted._symmetric is symmetric and converted._polys is polys
    assert all((type(value) is sp.Poly) is polys for value in converted)
    assert all(value.get_modulus() == 5 for value in converted.polys)
    assert converted.quotient_basis() == (1, y)
    assert converted.multiplication_matrix(y) == sp.Matrix([[0, -1 if symmetric else 4], [1, 0]])


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_nilpotent_directions_are_kept_in_the_complete_quotient(prime):
    x = sp.Symbol("field_nilpotent_x")
    basis = sp.groebner([x**3], x, modulus=prime)
    assert basis.quotient_basis() == (1, x, x*x)
    matrix = basis.multiplication_matrix(x)
    assert matrix == sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert matrix**3 == sp.zeros(3) and matrix**2 != sp.zeros(3)
    assert basis.multiplication_matrix(1) == sp.eye(3)
    assert basis.multiplication_matrix(0) == sp.zeros(3)


@pytest.mark.parametrize("order", ORDERS)
def test_complete_multiplication_columns_follow_declared_standard_monomials(order):
    x, y = sp.symbols("field_operator_x field_operator_y")
    basis = sp.groebner([x*x - 2, y*y - 3], x, y, modulus=5, order=order)
    monomials = basis.quotient_basis()
    assert monomials == (1, y, x, x*y)
    matrix = basis.multiplication_matrix(x + y)
    assert matrix == sp.Matrix([[0, -2, 2, 0], [1, 0, 0, 2],
                               [1, 0, 0, -2], [0, 1, 1, 0]])
    for column, monomial in enumerate(monomials):
        represented = sum((matrix[row, column]*term for row, term in enumerate(monomials)), sp.Integer(0))
        assert basis.contains((x+y)*monomial - represented)
    assert _residues(matrix, 5) == _residues(basis.multiplication_matrix(x) + basis.multiplication_matrix(y), 5)
    assert _residues(basis.multiplication_matrix(x)*basis.multiplication_matrix(y), 5) == _residues(basis.multiplication_matrix(x*y), 5)


@pytest.mark.parametrize("order", ORDERS)
def test_quotient_monomials_change_with_the_order_without_losing_multiplicity(order):
    x, y = sp.symbols("field_order_x field_order_y")
    basis = sp.groebner([x-y*y, y**3], x, y, modulus=3, order=order)
    assert basis.quotient_basis() == ((1, y, y*y) if order == "lex" else (1, y, x))
    assert basis.multiplication_matrix(y) == sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_characteristic_controls_ideal_dimension_and_operator_coefficients():
    x = sp.Symbol("field_characteristic_x")
    sources = [x*x + 1, x + 1]
    basis = sp.groebner(sources, x, modulus=2, order="grevlex").fglm("lex")
    assert basis.exprs == [x+1]
    assert basis.quotient_basis() == (1,)
    assert basis.multiplication_matrix(x) == sp.Matrix([[1]])
    assert sp.groebner(sources, x, domain="QQ").exprs == [1]
    # x^2+1 has no F_3 point, but its algebraic quotient still has dimension2.
    extension = sp.groebner([x*x+1], x, modulus=3)
    assert extension.quotient_basis() == (1, x)
    assert extension.multiplication_matrix(x) == sp.Matrix([[0, -1], [1, 0]])
    assert not extension.contains(x**3-x)


def test_changed_current_generators_are_used_for_quotient_operators():
    x = sp.Symbol("field_current_x")
    basis = sp.groebner([x], x, modulus=5)
    basis.exprs[:] = [0, 2*x*x-4, x**3-2*x, x*x-2]
    original = tuple(basis.exprs)
    assert basis.quotient_basis() == (1, x)
    assert basis.multiplication_matrix(x) == sp.Matrix([[0, 2], [1, 0]])
    assert tuple(basis.exprs) == original


@pytest.mark.parametrize("polys", [False, True])
def test_zero_quotient_and_explicit_constant_field_remain_usable(polys):
    x = sp.Symbol("field_constant_x")
    unit = sp.groebner([1], x, modulus=5, polys=polys)
    assert unit.quotient_basis() == ()
    assert unit.multiplication_matrix(x).shape == (0, 0)
    converted = unit.fglm("grlex")
    assert converted.exprs == [1] and converted.domain == "GF(5)"
    constant = sp.groebner([x], x, modulus=5, polys=polys).eliminate(x)
    assert constant.gens == () and constant.quotient_basis() == (1,)
    assert constant.multiplication_matrix(4) == sp.Matrix([[-1]])
    converted = constant.fglm("grevlex")
    assert converted.exprs == [] and converted.gens == () and converted.domain == "GF(5)"
    assert converted._polys is polys
    assert converted.quotient_basis() == (1,)


def test_positive_dimension_and_oversized_quotients_refuse_without_truncation():
    x, y = sp.symbols("field_bound_x field_bound_y")
    for sources in ([], [x*y]):
        basis = sp.groebner(sources, x, y, modulus=5)
        # A same-order no-op does not require finite dimension.
        assert basis.fglm("lex") is basis
        with pytest.raises(ValueError, match="finite-dimensional"):
            basis.fglm("grlex")
        with pytest.raises(ValueError, match="finite-dimensional"):
            basis.quotient_basis()
        with pytest.raises(ValueError, match="finite-dimensional"):
            basis.multiplication_matrix(0)
    large = sp.groebner([x**8, y**9], x, y, modulus=5)
    for operation in (large.quotient_basis, lambda: large.multiplication_matrix(0), lambda: large.fglm("grlex")):
        with pytest.raises(ValueError, match="dimension exceeds 64"):
            operation()


def test_same_order_fglm_rechecks_current_basis_instead_of_returning_unadmitted_self():
    x, y = sp.symbols("field_same_order_x field_same_order_y")
    basis = sp.groebner([x*y-1, y*y-x], x, y, modulus=5)
    basis.exprs[:] = [x*y-1, y*y-x]
    with pytest.raises(ValueError):
        basis.fglm("lex")
    with pytest.raises(ValueError):
        basis.fglm("invalid-order")


def test_typed_same_printed_symbols_survive_fglm_and_quotient_lifting():
    plain = sp.Symbol("field_typed")
    assumed = sp.Symbol("field_typed", positive=True)
    dummy = sp.Dummy("field_typed")
    axes = (dummy, assumed, plain)
    basis = sp.groebner([dummy-plain, assumed-plain, plain*plain-1], *axes,
                        modulus=5, order="grevlex")
    converted = basis.fglm("lex")
    assert all(actual is original for actual, original in zip(converted.gens, axes))
    monomials = converted.quotient_basis()
    assert monomials == (1, plain) and monomials[1] is plain
    assert converted.multiplication_matrix(assumed+dummy) == sp.Matrix([[0, 2], [2, 0]])


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
@pytest.mark.parametrize("shape", ["zero_product", "zero_power"])
def test_original_undefined_sources_survive_unit_and_zero_shortcuts(operation, shape):
    x = sp.Symbol("field_hidden_" + operation + "_" + shape)
    reciprocal = sp.Pow(x, -1, evaluate=False)
    hidden = (sp.Mul(0, reciprocal, evaluate=False) if shape == "zero_product"
              else sp.Pow(reciprocal, 0, evaluate=False))
    basis = sp.groebner([1], x, modulus=5)
    basis.exprs[:] = [1, hidden]
    with pytest.raises(ValueError):
        _operation(basis, operation, 0)
    clean = sp.groebner([1], x, modulus=5)
    with pytest.raises(ValueError):
        clean.multiplication_matrix(hidden)


@pytest.mark.parametrize("element", [sp.Rational(1, 2), sp.Float("1.0")])
def test_nonintegral_and_approximate_operator_sources_refuse(element):
    x = sp.Symbol("field_exact_source_x")
    for sources in ([x*x], [1]):
        basis = sp.groebner(sources, x, modulus=5)
        with pytest.raises((CoercionFailed, ValueError)):
            basis.multiplication_matrix(element)


def test_element_poly_field_and_original_declared_ring_are_retained():
    x, y = sp.symbols("field_poly_x field_poly_y")
    basis = sp.groebner([x*x, y], x, y, modulus=5)
    assert basis.multiplication_matrix(sp.Poly(x, x, modulus=5)) == sp.Matrix([[0, 0], [1, 0]])
    with pytest.raises(UnificationFailed):
        basis.multiplication_matrix(sp.Poly(x, x, modulus=7))
    forged = sp.Poly(0, x, modulus=5)
    forged._expr = sp.Add(y, -y, evaluate=False)
    with pytest.raises(ValueError):
        basis.multiplication_matrix(forged)


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
def test_unused_poly_generators_keep_their_immutable_native_identity(monkeypatch, operation):
    x = sp.Symbol("field_unused_" + operation)
    foreign = sp.Symbol("field_unused_" + operation, real=True)
    basis = sp.groebner([x], x, modulus=5)
    basis.exprs[:] = [x, sp.Poly(0, foreign, modulus=5)]
    monkeypatch.setattr(foreign, "_assumptions", dict(x._assumptions))
    with pytest.raises(ValueError, match="identity"):
        _operation(basis, operation, 0)


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
@pytest.mark.parametrize("change", ["expressions", "gens", "domain", "modulus", "symmetric",
                                    "polys", "held_children", "native_value", "assumptions"])
def test_original_source_and_controls_are_retained_through_native_work(monkeypatch, operation, change):
    x = sp.Symbol("field_mutation_" + operation + "_" + change)
    held = sp.Add(x*x, 1, evaluate=False)
    basis = sp.groebner([x*x+1], x, modulus=5)
    basis.exprs[:] = [held]
    bridge = _bridge(operation)
    original = getattr(_native, bridge)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "expressions":
            basis.exprs.append(x)
        elif change == "gens":
            basis.gens = ()
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
        elif change == "native_value":
            held._value = _native_expr(x)
        else:
            monkeypatch.setitem(x._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, bridge, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _operation(basis, operation, x)


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
def test_result_lifting_cannot_publish_after_source_mutation(monkeypatch, operation):
    from sympy.polys import _finitefield_groebner

    x = sp.Symbol("field_lift_" + operation)
    basis = sp.groebner([x*x+1], x, modulus=5)
    original = _finitefield_groebner._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        basis.exprs[:] = [1]
        return result

    monkeypatch.setattr(_finitefield_groebner, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _operation(basis, operation, x)


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
@pytest.mark.parametrize("failure", [ValueError, MemoryError])
def test_native_failure_is_terminal_without_field_change_or_budget_restart(monkeypatch, operation, failure):
    x = sp.Symbol("field_failure_" + operation)
    basis = sp.groebner([x*x+1], x, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure("original verifier budget exhausted or cancelled")

    def forbidden(*args, **kwargs):
        raise AssertionError("failed finite-quotient request restarted or entered QQ")

    monkeypatch.setattr(_native, _bridge(operation), refuse)
    for name in ("groebner_fglm_native", "quotient_multiplication_native",
                 "prime_field_groebner_native", "prime_field_groebner_zero_dimensional_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(failure, match="budget exhausted or cancelled"):
        _operation(basis, operation, x)
    assert calls == [1]


@pytest.mark.parametrize("operation", ["fglm", "quotient_basis", "multiplication_matrix"])
def test_inconsistent_retained_field_cannot_enter_rational_or_modular_work(monkeypatch, operation):
    x = sp.Symbol("field_metadata_" + operation)
    basis = sp.groebner([x*x+1], x, modulus=5)
    basis.domain = "QQ"

    def forbidden(*args, **kwargs):
        raise AssertionError("inconsistent field metadata reached mathematics")

    for name in (_bridge(operation), "groebner_fglm_native", "quotient_multiplication_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(DomainError):
        _operation(basis, operation, x)
