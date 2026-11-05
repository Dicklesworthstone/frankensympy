"""Complete quotient bases, multiplication operators and element relations."""

import pytest
import sympy as s
from sympy.core import _native
from sympy.polys.polytools import GroebnerBasis


def _basis(expressions, generators, order="lex", domain="QQ"):
    return GroebnerBasis(list(expressions), tuple(generators), domain, order)


def _equal(actual, expected):
    assert s.expand(actual - expected) == 0


def _forbid(*args, **kwargs):
    raise AssertionError("checked quotient operations must not use another generator")


def test_quadratic_quotient_basis_and_column_oriented_multiplication():
    x = s.Symbol("quotient_algebra_quadratic")
    ideal = _basis([x*x - 2], (x,))
    assert ideal.quotient_basis() == (1, x)
    assert ideal.multiplication_matrix(x) == s.Matrix([[0, 2], [1, 0]])
    assert ideal.multiplication_matrix(x + 1) == s.Matrix([[1, 2], [1, 1]])
    assert ideal.multiplication_matrix(1) == s.eye(2)
    assert ideal.multiplication_matrix(0) == s.zeros(2)


def test_nilpotent_multiplicity_survives_both_operator_and_minimal_polynomial():
    x, t = s.symbols("quotient_algebra_nilpotent quotient_algebra_parameter")
    ideal = _basis([x**3], (x,))
    assert ideal.quotient_basis() == (1, x, x*x)
    matrix = ideal.multiplication_matrix(x)
    assert matrix == s.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert matrix**3 == s.zeros(3)
    assert matrix**2 != s.zeros(3)
    _equal(ideal.minimal_polynomial(x, t), t**3)
    _equal(ideal.minpoly(x*x, t), t*t)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_multivariate_operator_acts_on_the_complete_quotient(order):
    x, y, t = s.symbols("quotient_algebra_x quotient_algebra_y quotient_algebra_t")
    ideal = _basis([x*x - 2, y*y - 3], (x, y), order)
    basis = ideal.quotient_basis()
    assert basis == (1, y, x, x*y)
    matrix = ideal.multiplication_matrix(x + y)
    assert matrix == s.Matrix([[0, 3, 2, 0], [1, 0, 0, 2],
                               [1, 0, 0, 3], [0, 1, 1, 0]])
    assert matrix == ideal.multiplication_matrix(x) + ideal.multiplication_matrix(y)
    for column, monomial in enumerate(basis):
        represented = sum((matrix[row, column]*term for row, term in enumerate(basis)), s.Integer(0))
        assert ideal.contains((x + y)*monomial - represented)
    _equal(ideal.minimal_polynomial(x + y, t), t**4 - 10*t*t + 1)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_quotient_basis_respects_the_declared_monomial_order(order):
    x, y = s.symbols("quotient_order_x quotient_order_y")
    ideal = _basis([x - y*y, y**3], (x, y), order)
    expected = (1, y, y*y) if order == "lex" else (1, y, x)
    assert ideal.quotient_basis() == expected
    assert ideal.multiplication_matrix(y) == s.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_current_generators_are_rechecked_without_assuming_they_form_a_basis():
    x, t = s.symbols("quotient_current_x quotient_current_t")
    ideal = _basis([x], (x,))
    ideal.exprs[:] = [0, 2*x*x - 4, x**3 - 2*x, x*x - 2]
    original = tuple(ideal.exprs)
    assert ideal.quotient_basis() == (1, x)
    assert ideal.multiplication_matrix(x) == s.Matrix([[0, 2], [1, 0]])
    _equal(ideal.minimal_polynomial(x, t), t*t - 2)
    assert tuple(ideal.exprs) == original


def test_integer_coefficients_extend_to_the_rational_quotient():
    x, t = s.symbols("quotient_integer_x quotient_integer_t")
    ideal = _basis([2*x + 1], (x,), domain="ZZ")
    assert ideal.quotient_basis() == (1,)
    assert ideal.multiplication_matrix(x) == s.Matrix([[-s.Rational(1, 2)]])
    _equal(ideal.minimal_polynomial(x, t), t + s.Rational(1, 2))
    assert ideal.domain == "ZZ"


def test_unit_ideal_has_zero_dimension_and_constant_annihilator_one():
    x, t = s.symbols("quotient_zero_ring_x quotient_zero_ring_t")
    for ideal in (_basis([1], (x,)), _basis([s.Rational(2, 3)], ())):
        assert ideal.quotient_basis() == ()
        assert ideal.multiplication_matrix(0).shape == (0, 0)
        assert ideal.multiplication_matrix(1) == s.Matrix(0, 0, [])
        assert ideal.minimal_polynomial(0, t) == 1
        assert ideal.minimal_polynomial(1, t) == 1


def test_constant_ring_qq_has_one_basis_vector_and_scalar_operators():
    t = s.Symbol("quotient_constant_t")
    ideal = _basis([], ())
    value = s.Rational(2, 3)
    assert ideal.quotient_basis() == (1,)
    assert ideal.multiplication_matrix(value) == s.Matrix([[value]])
    _equal(ideal.minimal_polynomial(value, t), t - value)
    _equal(ideal.minimal_polynomial(0, t), t)


@pytest.mark.parametrize("kind", ["quadratic_field", "multivariate_nilpotents", "constant_qq", "unit_ideal"])
def test_quotient_and_multiplication_matrix_minimal_polynomials_agree(kind):
    x, y, t = s.symbols("quotient_cross_x quotient_cross_y quotient_cross_t")
    ideal, element = {
        "quadratic_field": lambda: (_basis([x*x - 2], (x,)), x + 1),
        "multivariate_nilpotents": lambda: (_basis([x*x, y*y], (x, y)), x + y),
        "constant_qq": lambda: (_basis([], ()), s.Rational(2, 3)),
        "unit_ideal": lambda: (_basis([1], (x, y)), x + y),
    }[kind]()
    operator_polynomial = ideal.multiplication_matrix(element).minimal_polynomial(t)
    _equal(operator_polynomial.as_expr(), ideal.minimal_polynomial(element, t))


def test_positive_dimensional_quotient_distinguishes_algebraic_and_transcendental_elements():
    x, y, t = s.symbols("quotient_positive_x quotient_positive_y quotient_positive_t")
    ideal = _basis([x*x], (x, y))
    _equal(ideal.minimal_polynomial(x, t), t*t)
    _equal(ideal.minimal_polynomial(x + 1, t), (t - 1)**2)
    assert ideal.minimal_polynomial(y, t) is None
    assert ideal.minimal_polynomial(x + y, t) is None
    with pytest.raises(ValueError):
        ideal.quotient_basis()
    with pytest.raises(ValueError):
        ideal.multiplication_matrix(x)
    hyperbola = _basis([x*y - 1], (x, y))
    _equal(hyperbola.minimal_polynomial(x*y, t), t - 1)
    assert hyperbola.minimal_polynomial(x, t) is None


def test_minimal_polynomial_covers_every_component_not_one_selected_root():
    x, t = s.symbols("quotient_all_components_x quotient_all_components_t")
    ideal = _basis([x*(x - 1)], (x,))
    _equal(ideal.minimal_polynomial(x, t), t*(t - 1))
    assert ideal.minimal_polynomial(x, t) != t - 1
    assert ideal.minimal_polynomial(x, t) != t


def test_zero_ideal_proves_transcendence_without_conflating_constants():
    x, t = s.symbols("quotient_transcendental_x quotient_transcendental_t")
    ideal = _basis([], (x,))
    assert ideal.minimal_polynomial(x, t) is None
    _equal(ideal.minimal_polynomial(s.Rational(3, 5), t), t - s.Rational(3, 5))


def test_default_minimal_polynomial_variable_is_fresh_and_explicit_owner_is_restored():
    x = s.Symbol("quotient_parameter_identity")
    parameter = s.Symbol("quotient_parameter_identity", real=True)
    ideal = _basis([x*x - 2], (x,))
    explicit = ideal.minimal_polynomial(x, parameter)
    _equal(explicit, parameter*parameter - 2)
    assert len(explicit.free_symbols) == 1
    assert next(iter(explicit.free_symbols)) is parameter
    first, second = ideal.minimal_polynomial(x), ideal.minimal_polynomial(x)
    a, b = next(iter(first.free_symbols)), next(iter(second.free_symbols))
    assert type(a) is s.Dummy and type(b) is s.Dummy
    assert a is not b and a != b
    _equal(first, a*a - 2)
    _equal(second, b*b - 2)


def test_same_printed_typed_ring_variables_keep_their_original_owners():
    plain = s.Symbol("quotient_same_name")
    assumed = s.Symbol("quotient_same_name", positive=True)
    dummy = s.Dummy("quotient_same_name")
    generators = (dummy, assumed, plain)
    ideal = _basis([symbol**2 for symbol in generators], generators)
    monomials = ideal.quotient_basis()
    assert len(monomials) == 8
    assert all(any(symbol is original for original in generators)
               for monomial in monomials for symbol in monomial.free_symbols)
    assert all(any(monomial is original for monomial in monomials) for original in generators)
    t = s.Dummy("quotient_same_name")
    result = ideal.minimal_polynomial(dummy + assumed + plain, t)
    _equal(result, t**4)
    assert next(iter(result.free_symbols)) is t


def test_poly_elements_use_the_captured_expression_without_instance_callbacks(monkeypatch):
    x, t = s.symbols("quotient_poly_x quotient_poly_t")
    ideal = _basis([s.Poly(x*x - 2, x)], (x,))
    element = s.Poly(x + 1, x)
    assert ideal.multiplication_matrix(element) == s.Matrix([[1, 2], [1, 1]])
    _equal(ideal.minimal_polynomial(element, t), t*t - 2*t - 1)
    monkeypatch.setattr(element, "as_expr", _forbid)
    with pytest.raises(TypeError, match="overridden Poly.as_expr"):
        ideal.multiplication_matrix(element)
    with pytest.raises(TypeError, match="overridden Poly.as_expr"):
        ideal.minimal_polynomial(element, t)


@pytest.mark.parametrize("kind", ["ring_generator", "nonsymbol"])
def test_invalid_minimal_polynomial_parameter_refuses_before_generation(kind, monkeypatch):
    x = s.Symbol("quotient_parameter_refusal")
    ideal = _basis([x*x], (x,))
    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", _forbid)
    parameter = x if kind == "ring_generator" else x + 1
    with pytest.raises(ValueError if kind == "ring_generator" else TypeError):
        ideal.minimal_polynomial(x, parameter)


def test_preexisting_output_assumptions_match_the_native_atom_before_generation(monkeypatch):
    x = s.Symbol("quotient_stale_parameter_x")
    parameter = s.Symbol("quotient_stale_parameter_t", positive=True)
    ideal = _basis([x*x], (x,))
    monkeypatch.setitem(parameter._assumptions, "positive", False)
    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", _forbid)
    with pytest.raises(ValueError, match="inconsistent typed assumptions"):
        ideal.minimal_polynomial(x, parameter)


@pytest.mark.parametrize("method", ["multiplication_matrix", "minimal_polynomial"])
@pytest.mark.parametrize("kind", ["negative_power", "foreign_variable", "inexact"])
def test_elements_outside_the_exact_declared_polynomial_ring_refuse(method, kind):
    x = s.Symbol("quotient_admission_x")
    other = s.Symbol("quotient_admission_x", real=True)
    ideal = _basis([x*x - 2], (x,))
    element = {"negative_power": s.Pow(x, -1, evaluate=False),
               "foreign_variable": other, "inexact": s.Float(0.5)}[kind]
    with pytest.raises((TypeError, ValueError)):
        getattr(ideal, method)(element)


def test_operator_input_limits_and_stricter_elimination_limits(monkeypatch):
    generators = s.symbols("quotient_limit_0:8")
    ideal = _basis([*generators, *([0]*56)], generators)
    assert ideal.quotient_basis() == (1,)
    assert ideal.multiplication_matrix(generators[0]) == s.Matrix([[0]])
    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", _forbid)
    with pytest.raises(ValueError, match="seven variables and 63 ideal generators"):
        ideal.minimal_polynomial(generators[0])
    too_many_inputs = _basis([generators[0]]*64, (generators[0],))
    with pytest.raises(ValueError, match="seven variables and 63 ideal generators"):
        too_many_inputs.minimal_polynomial(0)


def test_oversized_quotient_dimension_is_not_truncated():
    x = s.Symbol("quotient_dimension_limit")
    ideal = _basis([x**65], (x,))
    with pytest.raises(ValueError):
        ideal.quotient_basis()
    with pytest.raises(ValueError):
        ideal.multiplication_matrix(x)


@pytest.mark.parametrize("method, native_name", [
    ("quotient_basis", "quotient_multiplication_native"),
    ("multiplication_matrix", "quotient_multiplication_native"),
    ("minimal_polynomial", "quotient_minimal_polynomial_native"),
])
@pytest.mark.parametrize("failure", [ValueError("quotient work exhausted"), NotImplementedError("quotient refusal")])
def test_native_operational_failures_are_terminal(method, native_name, failure, monkeypatch):
    x = s.Symbol("quotient_native_failure")
    ideal = _basis([x*x], (x,))
    calls = []

    def fail(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(_native, native_name, fail)
    monkeypatch.setattr(_native, "groebner_basis_native", _forbid)
    arguments = () if method == "quotient_basis" else (x,)
    with pytest.raises(type(failure)) as error:
        getattr(ideal, method)(*arguments)
    assert error.value is failure
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["quotient_basis", "multiplication_matrix", "minimal_polynomial"])
def test_mutation_during_lifting_prevents_result_publication(method, monkeypatch):
    import sympy.polys.polytools as polytools

    x = s.Symbol("quotient_lift_snapshot_" + method)
    ideal = _basis([x*x], (x,))
    lift = polytools._lift_native_result

    def mutate(*args):
        result = lift(*args)
        ideal.exprs.append(x)
        return result

    monkeypatch.setattr(polytools, "_lift_native_result", mutate)
    arguments = () if method == "quotient_basis" else (x,)
    with pytest.raises(RuntimeError, match="source mutated"):
        getattr(ideal, method)(*arguments)


def test_held_element_mutation_during_operator_generation_prevents_publication(monkeypatch):
    x = s.Symbol("quotient_element_snapshot")
    element = s.Add(x, 1, evaluate=False)
    ideal = _basis([x*x], (x,))
    native = _native.quotient_multiplication_native

    def mutate(*args):
        result = native(*args)
        s.Add.__dict__["_args"].__set__(element, (x, s.Integer(2)))
        return result

    monkeypatch.setattr(_native, "quotient_multiplication_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        ideal.multiplication_matrix(element)


def test_output_parameter_assumptions_are_bound_through_minimal_polynomial_generation(monkeypatch):
    x, t = s.symbols("quotient_parameter_snapshot_x quotient_parameter_snapshot_t")
    ideal = _basis([x*x], (x,))
    native = _native.quotient_minimal_polynomial_native

    def mutate(*args):
        result = native(*args)
        monkeypatch.setitem(t._assumptions, "positive", True)
        return result

    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        ideal.minimal_polynomial(x, t)


def test_proved_transcendence_is_not_published_after_source_mutation(monkeypatch):
    x, y, t = s.symbols("quotient_none_x quotient_none_y quotient_none_t")
    ideal = _basis([x*x], (x, y))
    native = _native.quotient_minimal_polynomial_native

    def mutate(*args):
        result = native(*args)
        assert result is None
        ideal.exprs.append(y)
        return result

    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        ideal.minimal_polynomial(y, t)
