"""Checked inverses and complete element relations over retained prime fields.

Univariate inverse fixtures and complete elimination ideals were independently
checked with the isolated SymPy 1.14.0 oracle. These extensions use the full
algebraic quotient, including nilpotents and positive-dimensional components.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotInvertible, UnificationFailed


ORDERS = ("lex", "grlex", "grevlex")
OPERATIONS = ("invert", "minimal_polynomial")


def _operation(basis, operation, element, parameter):
    return (basis.invert(element) if operation == "invert"
            else basis.minimal_polynomial(element, parameter))


def _bridge(operation):
    return ("prime_field_ideal_inverse_native" if operation == "invert"
            else "prime_field_quotient_minimal_polynomial_native")


@pytest.mark.parametrize("order", ORDERS)
def test_inverse_supports_positive_dimension_and_rechecks_current_generators(order):
    x, y = sp.symbols("field_inverse_positive_x field_inverse_positive_y")
    basis = sp.groebner([x], x, y, modulus=5, order=order)
    basis.exprs[:] = [0, 2*x*y-2, x*x*y-x, x*y-1]
    original = tuple(basis.exprs)
    assert basis.invert(x) == y
    assert basis.invert(y) == x
    assert tuple(basis.exprs) == original
    with pytest.raises(NotInvertible):
        basis.invert(x+y)


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_inverse_and_minpoly_return_expressions_with_retained_residue_policy(polys, symmetric):
    x, t = sp.symbols("field_function_policy_x field_function_policy_t")
    basis = sp.groebner([x*x-2], x, modulus=5, polys=polys, symmetric=symmetric)
    inverse = basis.invert(x)
    relation = basis.minpoly(x, t)
    assert inverse == (-2 if symmetric else 3)*x
    assert relation == t*t+(-2 if symmetric else 3)
    assert not isinstance(inverse, sp.Poly) and not isinstance(relation, sp.Poly)
    assert basis.domain == "GF(5)" and basis._polys is polys
    assert basis._symmetric is symmetric and basis.gens[0] is x
    assert basis.contains(x*inverse-1)


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_inverse_and_relations_preserve_nilpotents_in_every_characteristic(prime):
    x, t = sp.symbols("field_function_nilpotent_x field_function_nilpotent_t")
    basis = sp.groebner([x**3], x, modulus=prime)
    sign = 1 if prime == 2 else -1
    assert basis.invert(1+x) == 1+sign*x+x*x
    assert basis.minimal_polynomial(x, t) == t**3
    assert basis.minpoly(x*x, t) == t*t
    with pytest.raises(NotInvertible):
        basis.invert(x)
    with pytest.raises(NotInvertible):
        basis.invert(0)


def test_inseparable_relation_keeps_multiplicity_and_every_component():
    x, t = sp.symbols("field_inseparable_x field_inseparable_t")
    nilpotent = sp.groebner([x**3], x, modulus=3)
    assert nilpotent.minpoly(1+x, t) == t**3-1
    components = sp.groebner([x*x-x], x, modulus=3)
    assert components.minpoly(x, t) == t*t-t
    extension = sp.groebner([x*x+1], x, modulus=3)
    assert extension.minpoly(x, t) == t*t+1
    assert extension.invert(x) == -x
    assert not extension.contains(x**3-x)


@pytest.mark.parametrize("order", ORDERS)
def test_multivariate_inverse_and_relation_agree_with_complete_operator(order):
    x, y, t = sp.symbols("field_function_x field_function_y field_function_t")
    basis = sp.groebner([x*x-2, y*y-3], x, y, modulus=5, order=order)
    assert basis.minpoly(x+y, t) == t**4+1
    assert basis.invert(x+y) == y-x
    product = basis.multiplication_matrix(x+y)*basis.multiplication_matrix(y-x)
    residues = sp.Matrix(product.rows, product.cols, [int(value) % 5 for value in product])
    assert residues == sp.eye(4)


def test_positive_dimension_distinguishes_algebraic_elements_from_transcendence():
    x, y, t = sp.symbols("field_transcendental_x field_transcendental_y field_transcendental_t")
    basis = sp.groebner([x*x], x, y, modulus=5)
    assert basis.minpoly(x, t) == t*t
    assert basis.minpoly(x+1, t) == t*t-2*t+1
    assert basis.minpoly(y, t) is None
    assert basis.minpoly(x+y, t) is None
    laurent = sp.groebner([x*y-1], x, y, modulus=5)
    assert laurent.invert(x) == y
    assert laurent.minpoly(x, t) is None
    assert laurent.minpoly(x*y, t) == t-1
    free = sp.groebner([], x, modulus=5)
    assert free.minpoly(x, t) is None
    assert free.minpoly(2, t) == t-2
    assert free.invert(2) == -2


@pytest.mark.parametrize("polys", [False, True])
def test_zero_quotient_and_explicit_constant_field_have_exact_conclusions(polys):
    x, t = sp.symbols("field_function_constant_x field_function_constant_t")
    unit = sp.groebner([1], x, modulus=5, polys=polys)
    for element in (0, 1, x):
        assert unit.invert(element) == 0
        assert unit.minpoly(element, t) == 1
    constant = sp.groebner([x], x, modulus=5, polys=polys).eliminate(x)
    assert constant.gens == ()
    assert constant.invert(2) == -2
    assert constant.minpoly(2, t) == t-2
    assert constant.minpoly(0, t) == t
    with pytest.raises(NotInvertible):
        constant.invert(0)


@pytest.mark.parametrize("value", [0, 1])
def test_specialized_family_supports_fglm_operators_inverses_and_relations(value):
    x, y, a, t = sp.symbols("field_family_x field_family_y field_family_a field_family_t")
    sources = [sp.Poly(x*y-a, x, y, a, modulus=3),
               sp.Poly(y*y-x, x, y, a, modulus=3)]
    specialized = [source.eval(a, value) for source in sources]
    basis = sp.groebner(specialized, x, y, order="grevlex").fglm("lex")
    assert basis.domain == "GF(3)" and basis._polys is True
    assert basis.gens == (x, y) and basis.quotient_basis() == (1, y, y*y)
    assert basis.minpoly(y, t) == t**3-value
    matrix = basis.multiplication_matrix(y)
    assert matrix == sp.Matrix([[0, 0, value], [1, 0, 0], [0, 1, 0]])
    assert matrix**3 == value*sp.eye(3)
    if value:
        assert basis.invert(y) == y*y
    else:
        with pytest.raises(NotInvertible):
            basis.invert(y)


def test_parameter_and_ring_symbols_keep_distinct_typed_owners():
    plain = sp.Symbol("field_function_typed")
    assumed = sp.Symbol("field_function_typed", positive=True)
    dummy = sp.Dummy("field_function_typed")
    parameter = sp.Dummy("field_function_typed")
    basis = sp.groebner([dummy*assumed-1, plain-assumed], dummy, assumed, plain, modulus=5)
    inverse = basis.invert(dummy)
    assert inverse is plain
    assert basis.minpoly(dummy, parameter) is None
    finite = sp.groebner([plain*plain-2], plain, modulus=5)
    explicit = finite.minpoly(plain, assumed)
    assert explicit == assumed*assumed-2
    assert next(iter(explicit.free_symbols)) is assumed
    first, second = finite.minpoly(plain), finite.minimal_polynomial(plain)
    a, b = next(iter(first.free_symbols)), next(iter(second.free_symbols))
    assert type(a) is sp.Dummy and type(b) is sp.Dummy and a is not b and a != b
    assert first == a*a-2 and second == b*b-2


@pytest.mark.parametrize("domain", ["GF(5)", "QQ"])
@pytest.mark.parametrize("outcome", ["relation", "transcendence", "unit"])
@pytest.mark.parametrize("change", ["expressions", "held_children"])
def test_default_parameter_creation_retains_original_sources(monkeypatch, domain, outcome, change):
    x = sp.Symbol("field_default_parameter_"+domain+"_"+outcome+"_"+change)
    sources = [x*x] if outcome == "relation" else [1] if outcome == "unit" else []
    basis = sp.groebner(sources, x, domain=domain)
    element = sp.Add(x, 0, evaluate=False)
    replacement = x**3
    replacement_children = (x, sp.Integer(1))
    original = _native.py_symbol
    calls = []

    def mutate(*args, **kwargs):
        calls.append(1)
        if change == "expressions":
            basis.exprs[:] = [replacement]
        else:
            element._args = replacement_children
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("mutated original source reached minimal-polynomial generation")

    monkeypatch.setattr(_native, "py_symbol", mutate)
    monkeypatch.setattr(_native, "prime_field_quotient_minimal_polynomial_native", forbidden)
    monkeypatch.setattr(_native, "quotient_minimal_polynomial_native", forbidden)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.minpoly(element)
    assert calls == [1]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_poly_elements_keep_their_field_and_original_declared_ring(operation):
    x, y, t = sp.symbols("field_function_poly_x field_function_poly_y field_function_poly_t")
    basis = sp.groebner([x*x-2, y], x, y, modulus=5)
    element = sp.Poly(x, x, modulus=5)
    expected = -2*x if operation == "invert" else t*t-2
    assert _operation(basis, operation, element, t) == expected
    with pytest.raises(UnificationFailed):
        _operation(basis, operation, sp.Poly(x, x, modulus=7), t)
    forged = sp.Poly(0, x, modulus=5)
    forged._expr = sp.Add(y, -y, evaluate=False)
    with pytest.raises(ValueError):
        _operation(basis, operation, forged, t)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("shape", ["zero_product", "zero_power"])
def test_original_undefined_sources_cannot_hide_behind_unit_or_zero_shortcuts(operation, shape):
    x, t = sp.symbols("field_function_hidden_x field_function_hidden_t")
    reciprocal = sp.Pow(x, -1, evaluate=False)
    hidden = (sp.Mul(0, reciprocal, evaluate=False) if shape == "zero_product"
              else sp.Pow(reciprocal, 0, evaluate=False))
    basis = sp.groebner([1], x, modulus=5)
    with pytest.raises(ValueError):
        _operation(basis, operation, hidden, t)
    basis.exprs[:] = [1, hidden]
    with pytest.raises(ValueError):
        _operation(basis, operation, 0, t)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("element", [sp.Rational(1, 2), sp.Float("1.0")])
def test_nonintegral_or_approximate_sources_refuse_even_in_zero_quotient(operation, element):
    x, t = sp.symbols("field_function_exact_x field_function_exact_t")
    basis = sp.groebner([1], x, modulus=5)
    with pytest.raises((CoercionFailed, ValueError)):
        _operation(basis, operation, element, t)


@pytest.mark.parametrize("kind", ["ring_generator", "expression", "stale_assumptions"])
def test_invalid_output_parameter_refuses_before_native_generation(monkeypatch, kind):
    x = sp.Symbol("field_function_parameter_x")
    parameter = sp.Symbol("field_function_parameter_t", positive=True)
    basis = sp.groebner([x*x], x, modulus=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid output parameter reached native generation")

    monkeypatch.setattr(_native, _bridge("minimal_polynomial"), forbidden)
    if kind == "ring_generator":
        parameter = x
    elif kind == "expression":
        parameter = x+1
    else:
        monkeypatch.setitem(parameter._assumptions, "positive", False)
    with pytest.raises(TypeError if kind == "expression" else ValueError):
        basis.minpoly(x, parameter)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("change", ["expressions", "gens", "domain", "modulus", "symmetric",
                                    "polys", "held_children", "native_value", "assumptions"])
def test_original_sources_and_metadata_are_retained_through_native_work(monkeypatch, operation, change):
    x, t = sp.symbols("field_function_mutation_x_"+operation+"_"+change+" field_function_mutation_t")
    held = sp.Add(x*x, 1, evaluate=False)
    basis = sp.groebner([x*x+1], x, modulus=5)
    basis.exprs[:] = [held]
    original = getattr(_native, _bridge(operation))

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

    monkeypatch.setattr(_native, _bridge(operation), mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _operation(basis, operation, x, t)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_mutation_prevents_nonunit_or_transcendence_conclusions(monkeypatch, operation):
    x, t = sp.symbols("field_function_none_x field_function_none_t")
    basis = sp.groebner([], x, modulus=5)
    original = getattr(_native, _bridge(operation))

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is None
        basis.exprs.append(x)
        return result

    monkeypatch.setattr(_native, _bridge(operation), mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _operation(basis, operation, x, t)


@pytest.mark.parametrize("outcome", ["relation", "transcendence", "unit"])
def test_output_parameter_is_retained_even_for_scalar_or_none_results(monkeypatch, outcome):
    x = sp.Symbol("field_parameter_retained_x_"+outcome)
    t = sp.Symbol("field_parameter_retained_t_"+outcome)
    sources = [x*x] if outcome == "relation" else [1] if outcome == "unit" else []
    basis = sp.groebner(sources, x, modulus=5)
    original = _native.prime_field_quotient_minimal_polynomial_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setitem(t._assumptions, "real", True)
        return result

    monkeypatch.setattr(_native, "prime_field_quotient_minimal_polynomial_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        basis.minpoly(x, t)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_result_lifting_cannot_publish_after_source_mutation(monkeypatch, operation):
    from sympy.polys import _finitefield_groebner

    x, t = sp.symbols("field_function_lift_x field_function_lift_t")
    basis = sp.groebner([x*x+1], x, modulus=5)
    original = _finitefield_groebner._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        basis.exprs[:] = [1]
        return result

    monkeypatch.setattr(_finitefield_groebner, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _operation(basis, operation, x, t)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("failure", [ValueError, MemoryError])
def test_native_failure_is_terminal_without_budget_restart_or_rational_fallback(monkeypatch, operation, failure):
    x, t = sp.symbols("field_function_failure_x field_function_failure_t")
    basis = sp.groebner([x*x+1], x, modulus=5)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure("original verifier budget exhausted or cancelled")

    def forbidden(*args, **kwargs):
        raise AssertionError("failed prime-field quotient request restarted or entered QQ")

    monkeypatch.setattr(_native, _bridge(operation), refuse)
    for name in ("ideal_inverse_native", "quotient_minimal_polynomial_native",
                 "prime_field_groebner_native", "prime_field_groebner_zero_dimensional_native"):
        monkeypatch.setattr(_native, name, forbidden)
    with pytest.raises(failure, match="budget exhausted or cancelled"):
        _operation(basis, operation, x, t)
    assert calls == [1]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_inconsistent_metadata_and_oversized_requests_refuse_before_mathematics(monkeypatch, operation):
    x, t = sp.symbols("field_function_bounds_x field_function_bounds_t")
    basis = sp.groebner([x*x], x, modulus=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("inadmissible request reached native generation")

    monkeypatch.setattr(_native, _bridge(operation), forbidden)
    basis.exprs[:] = [0]*32
    with pytest.raises(ValueError, match="31 ideal generators"):
        _operation(basis, operation, 0, t)
    basis.exprs[:] = [1]
    basis.domain = "QQ"
    with pytest.raises(DomainError):
        _operation(basis, operation, 0, t)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_raw_bridge_admits_all_sources_and_rejects_malformed_domains(operation):
    x, t = sp.symbols("field_function_raw_x field_function_raw_t")
    bridge = getattr(_native, _bridge(operation))
    bindings = [_symbol_binding(x)]

    def call(sources, prime=5, variables=None, parameter=t):
        arguments = (sources, bindings if variables is None else variables, prime)
        return (bridge(*arguments) if operation == "invert"
                else bridge(*arguments, _symbol_binding(parameter)))

    with pytest.raises(ValueError):
        call([])
    with pytest.raises(ValueError):
        call([_native_expr(0)]*33)
    with pytest.raises(ValueError):
        call([_native_expr(0), _native_expr(1)], prime=9)
    with pytest.raises(ValueError):
        call([_native_expr(0), _native_expr(1)], variables=bindings*2)
    with pytest.raises(ValueError):
        call([_native_expr(0), _native_expr(1), _native_expr(sp.sin(x))])
    if operation == "minimal_polynomial":
        with pytest.raises(ValueError):
            call([_native_expr(0), _native_expr(1)], parameter=x)
