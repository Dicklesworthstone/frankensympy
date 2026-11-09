"""Source-bound prime-field evaluation through the actual native engine.

The exact coefficient, container and generator-index conventions below were
checked separately against SymPy 1.14.0. Refusal fixtures preserve the existing
bounded exact-integer domain rather than enabling an approximate fallback.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import _finitefield
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, UnificationFailed


def fixture(cls=sp.Poly, symmetric=True):
    x, y, z = sp.symbols("x y z")
    return cls(x**2*y + 2*x*y**2 + 3*z + 4, x, y, z, modulus=5,
               symmetric=symmetric), (x, y, z)


@pytest.mark.parametrize("selector", ["default", "symbol", "index"])
def test_partial_evaluation_removes_only_the_selected_generator(selector):
    polynomial, (x, y, z) = fixture()
    result = (polynomial.eval(2) if selector == "default" else
              polynomial.eval(x if selector == "symbol" else 0, 2))
    assert type(result) is sp.Poly
    assert result.gens == (y, z)
    assert result.gens[0] is y and result.gens[1] is z
    assert result.as_dict() == {(2, 0): -1, (1, 0): -1, (0, 1): -2, (0, 0): -1}
    assert result.get_modulus() == 5
    assert str(result.domain) == "GF(5)"


def test_middle_and_negative_indices_preserve_original_ring_order():
    polynomial, (x, y, z) = fixture()
    middle = polynomial.eval(y, -1)
    assert middle.gens == (x, z)
    assert middle.as_dict() == {(2, 0): -1, (1, 0): 2, (0, 1): -2, (0, 0): -1}
    last = polynomial.eval(-1, 8)
    assert last.gens == (x, y)
    assert last.as_dict() == {(2, 1): 1, (1, 2): 2, (0, 0): -2}


@pytest.mark.parametrize("container", [list, tuple])
def test_partial_and_full_positional_evaluation(container):
    polynomial, (_, _, z) = fixture()
    partial = polynomial.eval(container([2, 3]))
    assert partial.gens == (z,)
    assert partial.as_dict() == {(1,): -2, (0,): 2}
    assert polynomial.eval(container([2, 3, 4])) == -1


def test_ordered_mapping_matches_sequential_public_generator_indices():
    x, y, z = sp.symbols("x y z")
    polynomial = sp.Poly(x + 2*y + 3*z, x, y, z, modulus=5)
    # After assigning x, index 1 selects z in the remaining (y,z) ring.
    result = polynomial.eval({0: 2, 1: 3})
    assert result.gens == (y,)
    assert result.as_dict() == {(1,): 2, (0,): 1}
    result = polynomial.eval({1: 2, 0: 3})
    assert result.gens == (z,)
    assert result.as_dict() == {(1,): -2, (0,): 2}
    result = polynomial.eval({-1: 2, -2: 3})
    assert result.gens == (y,)
    assert result.as_dict() == {(1,): 2, (0,): -1}
    assert polynomial.eval({z: 2, x: 3, y: 4}) == 2
    with pytest.raises(GeneratorsError):
        polynomial.eval({0: 2, x: 3})


@pytest.mark.parametrize("value", [0, 1, -1, 2, -(1 << 200) - 3, 1 << 300])
@pytest.mark.parametrize("wrap", [int, sp.Integer])
def test_univariate_integer_values_keep_exact_field_semantics(value, wrap):
    x = sp.Symbol("x")
    polynomial = sp.Poly(2*x**3 + 3*x + 4, x, modulus=5)
    expected = (2*value**3 + 3*value + 4) % 5
    if expected > 2:
        expected -= 5
    assert polynomial.eval(wrap(value)) == expected


@pytest.mark.parametrize("value", [0, 1, -1, 42])
def test_integer_shell_atoms_accept_equivalent_native_integral_rational_storage(monkeypatch, value):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x + 2, x, modulus=5)
    integer = sp.Integer(value)
    monkeypatch.setattr(integer, "_value", _native.py_rational(value, 1))
    expected = (value + 2) % 5
    assert polynomial.eval(integer) == (expected - 5 if expected > 2 else expected)


def test_purepoly_and_unsigned_lifts_keep_their_public_metadata():
    polynomial, (_, y, z) = fixture(sp.PurePoly, symmetric=False)
    result = polynomial.eval(2)
    assert type(result) is sp.PurePoly
    assert result.gens == (y, z)
    assert result.as_dict() == {(2, 0): 4, (1, 0): 4, (0, 1): 3, (0, 0): 4}
    assert result.get_modulus() == 5
    assert polynomial.eval([2, 3, 4]) == 4


@pytest.mark.parametrize("empty", [[], (), {}])
def test_empty_assignment_preserves_identity_after_one_checked_call(monkeypatch, empty):
    polynomial, _ = fixture()
    original = _native.prime_field_evaluate_native
    calls = []

    def checked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_evaluate_native", checked)
    assert polynomial.eval(empty) is polynomial
    assert calls == [1]


def test_zero_and_unused_generators_are_not_reinferred_after_evaluation():
    x = sp.Symbol("equal_name", positive=True)
    other_x = sp.Symbol("equal_name", negative=True)
    unused = sp.Symbol("unused")
    polynomial = sp.Poly(x*other_x + 1, x, other_x, unused, modulus=5)
    result = polynomial.eval(x, 0)
    assert result.gens == (other_x, unused)
    assert result.gens[0] is other_x and result.gens[1] is unused
    assert result.as_dict() == {(0, 0): 1}
    zero = sp.Poly(0, x, other_x, unused, modulus=5).eval(x, 0)
    assert zero.gens == (other_x, unused) and zero.is_zero
    assert zero.eval([3, 4]) == 0
    with pytest.raises(GeneratorsError):
        polynomial.eval(sp.Symbol("equal_name"), 2)


def test_exact_ring_contraction_is_not_interpolation_over_field_points():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x**2 + x + y, x, y, modulus=2)
    remaining = polynomial.eval(y, 0)
    assert remaining.gens == (x,)
    assert remaining.as_dict() == {(2,): 1, (1,): 1}
    assert remaining.eval(0) == remaining.eval(1) == 0
    assert not remaining.is_zero


@pytest.mark.parametrize("parameter_value", [0, 1])
def test_specialized_polynomials_feed_checked_order_conversion_and_quotient_algebra(parameter_value):
    x, y, parameter = sp.symbols("x y a")
    sources = [
        sp.Poly(x*y - parameter, x, y, parameter, modulus=3),
        sp.Poly(y**2 - x, x, y, parameter, modulus=3),
    ]
    specialized = [source.eval(parameter, parameter_value) for source in sources]
    assert all(poly.gens == (x, y) and poly.get_modulus() == 3 for poly in specialized)

    # The field is inferred from the specialized Poly objects. Keeping their
    # Poly output mode exercises metadata preservation across both bridges.
    basis = sp.groebner(specialized, x, y, order="grevlex")
    lex = basis.fglm("lex")
    assert str(basis.domain) == str(lex.domain) == "GF(3)"
    assert lex.gens == (x, y)
    assert list(lex) == [
        sp.Poly(x - y**2, x, y, modulus=3),
        sp.Poly(y**3 - parameter_value, x, y, modulus=3),
    ]
    assert all(type(poly) is sp.Poly and poly.get_modulus() == 3 for poly in lex)
    assert lex.quotient_basis() == (sp.Integer(1), y, y**2)
    assert lex.multiplication_matrix(y) == sp.Matrix([
        [0, 0, parameter_value],
        [1, 0, 0],
        [0, 1, 0],
    ])
    # Both fibers have multiplicity three: y^3 and (y-1)^3 over GF(3).
    # Evaluating at field points or taking a radical would lose directions.
    assert len(lex.quotient_basis()) == 3


def test_multi_assignment_has_one_native_call_and_no_univariate_or_qq_fallback(monkeypatch):
    polynomial, (x, y, z) = fixture()
    original = _native.prime_field_evaluate_native
    calls = []

    def checked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation fell back to an unrelated arithmetic lane")

    monkeypatch.setattr(_native, "prime_field_evaluate_native", checked)
    monkeypatch.setattr(_native, "prime_field_polynomial_native", forbidden)
    monkeypatch.setattr(_native, "poly_multivariate_div_rem_native", forbidden)
    assert polynomial.eval({z: 4, y: 3, x: 2}) == -1
    assert calls == [1]


@pytest.mark.parametrize("invalid", [sp.Rational(1, 2), sp.Float(1), 1.0, True, sp.Symbol("value")])
def test_values_cannot_silently_change_the_exact_domain(monkeypatch, invalid):
    polynomial, _ = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported source reached the native evaluator")

    monkeypatch.setattr(_native, "prime_field_evaluate_native", forbidden)
    with pytest.raises(UnificationFailed):
        polynomial.eval(invalid)


def test_malformed_generators_value_counts_and_custom_controls_fail_before_work():
    polynomial, (x, y, z) = fixture()
    for selector in (-4, 3, sp.Symbol("w"), object()):
        with pytest.raises(GeneratorsError):
            polynomial.eval(selector, 2)
    for values in ([1, 2, 3, 4], (1, 2, 3, 4), {x: 1, y: 2, z: 3, 0: 4}):
        with pytest.raises(ValueError, match="too many"):
            polynomial.eval(values)
    with pytest.raises(TypeError):
        polynomial.eval()
    with pytest.raises(TypeError):
        polynomial.eval(x, 1, 2)

    class Value:
        def __int__(self):
            raise AssertionError("custom coercion executed")

    with pytest.raises(UnificationFailed):
        polynomial.eval(Value())


def test_large_original_integer_values_refuse_even_when_source_is_zero():
    x, y = sp.symbols("x y")
    zero = sp.Poly(0, x, y, modulus=5)
    with pytest.raises(ValueError, match="4096"):
        zero.eval(1 << 4096)


@pytest.mark.parametrize("field,value", [("_modulus", 7), ("_gens", ()), ("_symmetric", False)])
def test_source_mutation_after_native_work_prevents_publication(monkeypatch, field, value):
    polynomial, _ = fixture()
    original = _native.prime_field_evaluate_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(polynomial, field, value)
        return result

    monkeypatch.setattr(_native, "prime_field_evaluate_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial.eval(2)


def test_sequence_is_retained_before_scalar_admission(monkeypatch):
    polynomial, _ = fixture()
    values = [sp.Integer(2), sp.Integer(3)]
    original = _native.prime_field_source_preflight_native

    def mutate(*args, **kwargs):
        values[0] = sp.Integer(4)
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "prime_field_source_preflight_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial.eval(values)


def test_mapping_remains_bound_through_result_lifting(monkeypatch):
    polynomial, (x, _, z) = fixture()
    values = {x: 2}
    original = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        values[z] = 3
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mapping mutated"):
        polynomial.eval(values)


def test_empty_assignment_still_rejects_invalid_retained_source():
    polynomial, (x, _, _) = fixture()
    polynomial._expr = sp.sin(x)
    with pytest.raises((ValueError, NotImplementedError)):
        polynomial.eval({})


def test_admitted_native_failures_are_terminal(monkeypatch):
    polynomial, _ = fixture()
    failure = ValueError("evaluation compute budget exhausted")

    def exhausted(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_native, "prime_field_evaluate_native", exhausted)
    with pytest.raises(ValueError) as raised:
        polynomial.eval([2, 3])
    assert raised.value is failure


def test_raw_bridge_bounds_all_sources_before_zero_assignment():
    x, y = sp.symbols("x y")
    variables = [_symbol_binding(x), _symbol_binding(y)]
    values = [_native_expr(0), None]
    for source in (
        sp.Mul(0, sp.sin(y), evaluate=False),
        sp.Pow(sp.sin(y), 0, evaluate=False),
        sp.Mul(5, sp.Pow(x, 65, evaluate=False), evaluate=False),
    ):
        with pytest.raises(ValueError):
            _native.prime_field_evaluate_native(_native_expr(source), variables, 5, values)
    for invalid in (_native_expr(sp.Rational(1, 2)), _native_expr(1 << 4096), _native_expr(x)):
        with pytest.raises(ValueError):
            _native.prime_field_evaluate_native(_native_expr(0), variables, 5, [invalid, None])
    with pytest.raises(ValueError):
        _native.prime_field_evaluate_native(_native_expr(0), variables, 5, [None])
    # Direct native integral rational atoms retain their exact value.
    assert _native.prime_field_evaluate_native(
        _native_expr(x), variables, 5, [_native.py_rational(-6, 1), None]).exact_numerator() == -1

