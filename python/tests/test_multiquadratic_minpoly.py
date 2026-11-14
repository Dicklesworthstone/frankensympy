"""Exact multiquadratic public results; expected coefficients use SymPy 1.14.0."""

import pytest
import sympy as s

from sympy.polys import numberfields as nf
from sympy.polys.polyerrors import PolynomialError


def _held_root(value):
    return s.Pow(s.sympify(value), s.Rational(1, 2), evaluate=False)


def _assert_polynomial(value, expected):
    x = s.Symbol("minpoly_variable")
    actual = s.minpoly(value, x)
    assert not isinstance(actual, s.Poly)
    assert s.Poly(actual, x).all_coeffs() == [s.Integer(coefficient) for coefficient in expected]
    polynomial = s.minimal_polynomial(value, x, polys=True)
    assert type(polynomial) is s.Poly
    assert polynomial.gens[0] is x
    assert polynomial.domain == "QQ"
    assert polynomial.as_expr() == actual
    return polynomial


@pytest.mark.parametrize("value,expected", [
    (0, [1, 0]),
    (7, [1, -7]),
    (s.Rational(2, 3), [3, -2]),
    (s.Rational(-5, 7), [7, 5]),
    (s.sqrt(2), [1, 0, -2]),
    (2 + 3*s.sqrt(2), [1, -4, -14]),
    (s.sqrt(s.Rational(2, 3)), [3, 0, -2]),
    (s.Rational(2, 3) + s.sqrt(s.Rational(2, 3)), [9, -12, -2]),
    (s.sqrt(2) + s.sqrt(3), [1, 0, -10, 0, 1]),
    (1/(s.sqrt(2) + s.sqrt(3)), [1, 0, -10, 0, 1]),
    (1/(1 + s.sqrt(2) + s.sqrt(3)), [8, -16, 4, 4, -1]),
    ((s.sqrt(2) + s.sqrt(3))**3, [1, 0, -970, 0, 1]),
    (s.sqrt(2) + s.sqrt(3) + s.sqrt(6), [1, 0, -22, -48, -23]),
    (s.sqrt(2) + s.sqrt(3) + s.sqrt(5), [1, 0, -40, 0, 352, 0, -960, 0, 576]),
    (s.I, [1, 0, 1]),
    (s.sqrt(2) + s.I, [1, 0, -2, 0, 9]),
    (s.sqrt(s.Rational(2, 3)) + s.sqrt(s.Rational(3, 2)), [6, 0, -25]),
])
def test_exact_public_polynomials(value, expected):
    _assert_polynomial(value, expected)


def test_four_independent_square_classes_reach_degree_sixteen():
    value = s.sqrt(2) + s.sqrt(3) + s.sqrt(5) + s.sqrt(7)
    _assert_polynomial(value, [
        1, 0, -136, 0, 6476, 0, -141912, 0, 1513334, 0,
        -7453176, 0, 13950764, 0, -5596840, 0, 46225,
    ])


def test_dependent_square_classes_do_not_inflate_the_minimal_degree():
    root2, root3, root8 = (_held_root(value) for value in (2, 3, 8))
    _assert_polynomial(s.Add(root2, root8, evaluate=False), [1, 0, -18])
    _assert_polynomial(s.Mul(root2, root8, evaluate=False), [1, -4])
    value = s.Add(s.Mul(root2, root2, evaluate=False),
                  s.Mul(root3, root3, evaluate=False), evaluate=False)
    _assert_polynomial(value, [1, -5])


def test_held_rational_square_roots_have_minimal_linear_polynomials():
    _assert_polynomial(_held_root(4), [1, -2])
    _assert_polynomial(_held_root(s.Rational(1, 9)), [3, -1])
    _assert_polynomial(_held_root(0), [1, 0])
    _assert_polynomial(s.Pow(0, 0, evaluate=False), [1, -1])


def test_dependent_negative_radicands_keep_the_principal_root_phase():
    root2, root3, root8 = (_held_root(value) for value in (-2, -3, -8))
    _assert_polynomial(s.Add(root2, root8, evaluate=False), [1, 0, 18])
    ratio = s.Mul(root8, s.Pow(root2, -1, evaluate=False), evaluate=False)
    _assert_polynomial(ratio, [1, -2])
    opposite = s.Add(s.Mul(root2, root3, evaluate=False), _held_root(6), evaluate=False)
    _assert_polynomial(opposite, [1, 0])


def test_negative_root_phase_with_three_negative_basis_generators():
    roots = [_held_root(value) for value in (-2, -3, -5)]
    product = s.Mul(*roots, evaluate=False)
    # Product = -I*sqrt(30), whereas the principal sqrt(-30) is +I*sqrt(30).
    _assert_polynomial(s.Add(product, _held_root(-30), evaluate=False), [1, 0])


def test_reciprocal_and_negative_half_integer_powers():
    _assert_polynomial(s.Pow(s.Integer(2), s.Rational(-3, 2), evaluate=False), [8, 0, -1])
    _assert_polynomial(s.Pow(s.Integer(-2), s.Rational(3, 2), evaluate=False), [1, 0, 8])


def test_default_generator_and_public_module_routes():
    first = s.minpoly(s.sqrt(2), polys=True)
    second = s.minimal_polynomial(s.sqrt(2), polys=True)
    assert type(first) is s.PurePoly
    assert type(second) is s.PurePoly
    assert type(first.gens[0]) is s.Dummy
    assert first.gens[0] != second.gens[0]
    assert first.all_coeffs() == second.all_coeffs() == [1, 0, -2]
    from sympy.polys import minpoly, minimal_polynomial
    assert minpoly is nf.minpoly is s.minpoly
    assert minimal_polynomial is nf.minimal_polynomial is s.minimal_polynomial
    assert not isinstance(s.minpoly(s.sqrt(2)), s.Poly)


@pytest.mark.parametrize("generator", [s.Symbol("same", positive=True), s.Dummy("same")])
def test_explicit_generator_identity_and_options(generator):
    for compose in (True, False):
        polynomial = s.minpoly(1/s.sqrt(2), generator, compose=compose, polys=True, domain="QQ")
        assert type(polynomial) is s.Poly
        assert polynomial.gens[0] is generator
        assert polynomial.all_coeffs() == [2, 0, -1]
        assert next(iter(polynomial.as_expr().free_symbols)) is generator


def test_exact_polynomial_annihilation_in_original_radicals():
    x = s.Symbol("annihilation_x")
    for value in (s.sqrt(2) + s.sqrt(3), s.sqrt(2) + s.I, 1 + s.sqrt(2)):
        polynomial = s.minpoly(value, x)
        assert s.simplify(s.expand(polynomial.subs(x, value))) == 0


@pytest.mark.parametrize("outer", ["identity", "zero_product", "zero_power", "cancellation"])
def test_held_division_by_zero_is_admitted_before_outer_shortcuts(outer):
    invalid = s.Pow(s.Integer(0), -1, evaluate=False)
    value = {
        "identity": lambda: invalid,
        "zero_product": lambda: s.Mul(0, invalid, evaluate=False),
        "zero_power": lambda: s.Pow(invalid, 0, evaluate=False),
        "cancellation": lambda: s.Add(invalid, s.Mul(-1, invalid, evaluate=False), evaluate=False),
    }[outer]()
    with pytest.raises(ZeroDivisionError):
        s.minpoly(value)


def test_exact_algebraic_zero_denominator_is_not_erased():
    root = _held_root(2)
    zero = s.Add(s.Mul(root, root, evaluate=False), -2, evaluate=False)
    denominator = s.Pow(zero, -1, evaluate=False)
    with pytest.raises(ZeroDivisionError):
        s.minpoly(s.Mul(0, denominator, evaluate=False))


@pytest.mark.parametrize("factory", [
    lambda: s.Symbol("parameter"),
    lambda: s.sqrt(s.Symbol("parameter")),
    lambda: s.pi,
    lambda: s.Float("0.5"),
    lambda: s.zoo,
    lambda: s.nan,
    lambda: s.sqrt(1 + s.sqrt(2)),
    lambda: s.Pow(s.Integer(2), s.Rational(1, 3), evaluate=False),
])
def test_unsupported_domains_and_extensions_refuse_explicitly(factory):
    with pytest.raises(NotImplementedError):
        s.minpoly(factory())


def test_custom_objects_are_not_coerced_or_inspected():
    calls = []

    class Foreign:
        def _sympy_(self):
            calls.append("conversion")
            return s.sqrt(2)

    class ForeignPow(s.Pow):
        @property
        def args(self):
            calls.append("args")
            return (s.Integer(2), s.Rational(1, 2))

    for value in (Foreign(), object.__new__(ForeignPow)):
        with pytest.raises(NotImplementedError):
            s.minpoly(value)
    assert calls == []


def test_source_cycles_and_retained_native_disagreement_refuse():
    value = s.Add(_held_root(2), 1, evaluate=False)
    value._args = (value,)
    with pytest.raises(ValueError, match="cycle"):
        s.minpoly(value)
    value = s.Add(_held_root(2), 1, evaluate=False)
    value._args = (s.Integer(7), s.Integer(8))
    with pytest.raises(ValueError, match="arguments disagree"):
        s.minpoly(value)


def test_source_mutation_during_checked_native_phase_is_terminal(monkeypatch):
    value = s.Add(_held_root(2), 1, evaluate=False)
    original = nf._checked_operator_polynomial

    def mutate(rows, work):
        result = original(rows, work)
        value._args = (s.Integer(99), s.Integer(100))
        return result

    monkeypatch.setattr(nf, "_checked_operator_polynomial", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.minpoly(value)


def test_checked_matrix_refusal_does_not_fall_back(monkeypatch):
    calls = []

    def refuse(rows, work):
        calls.append(len(rows))
        raise ValueError("checked matrix refusal")

    monkeypatch.setattr(nf, "_checked_operator_polynomial", refuse)
    with pytest.raises(ValueError, match="checked matrix refusal"):
        s.minpoly(s.sqrt(2) + s.sqrt(3))
    assert calls == [4]


def test_mutated_native_constructor_names_cannot_change_the_checked_operator(monkeypatch):
    from sympy.core import _native

    value = s.sqrt(2) + s.sqrt(3)
    x = s.Symbol("frozen_bridge_x")
    original = nf._checked_operator_polynomial
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("mutable module constructor must not replace the admitted bridge")

    def mutate(rows, work):
        monkeypatch.setattr(_native, "py_rational", forbidden)
        monkeypatch.setattr(_native, "py_integer", forbidden)
        return original(rows, work)

    monkeypatch.setattr(nf, "_checked_operator_polynomial", mutate)
    with pytest.raises(NotImplementedError, match="original native bridge"):
        s.minpoly(value, x)
    assert calls == []


def test_mutated_scalar_descriptors_are_rejected_before_hooks(monkeypatch):
    x = s.Symbol("descriptor_x")
    calls = []

    def forbidden(self):
        calls.append("name")
        raise AssertionError("user property must not run")

    monkeypatch.setattr(s.Symbol, "name", property(forbidden))
    with pytest.raises(NotImplementedError):
        s.minpoly(s.sqrt(2), x)
    assert calls == []


def test_field_degree_work_power_and_coefficient_limits(monkeypatch):
    with pytest.raises(ValueError, match="four independent"):
        s.minpoly(s.Add(*(_held_root(q) for q in (2, 3, 5, 7, 11)), evaluate=False))
    with pytest.raises(ValueError, match="power exceeds"):
        s.minpoly(s.Pow(_held_root(2), 1025, evaluate=False))
    with pytest.raises(ValueError, match="4096 bits"):
        s.minpoly(1 << 4096)
    monkeypatch.setattr(nf, "_MAX_WORK", 3)
    with pytest.raises(ValueError, match="work limit"):
        s.minpoly(s.sqrt(2) + s.sqrt(3))


def test_shared_held_owners_cannot_hide_an_oversized_native_tree():
    value = s.Integer(1)
    for _ in range(12):
        value = s.Add(value, value, evaluate=False)
    # Thirteen retained owners represent 8191 native tree nodes. Borrowed
    # admission must refuse before `.args` copies that expanded payload.
    with pytest.raises(ValueError, match="node"):
        s.minpoly(value)


def test_invalid_options_and_noncommutative_generator_refuse():
    for kwargs in ({"compose": "arbitrary"}, {"polys": "yes"}):
        with pytest.raises(TypeError):
            s.minpoly(s.sqrt(2), **kwargs)
    with pytest.raises(NotImplementedError, match="rational domain"):
        s.minpoly(s.sqrt(2), domain="ZZ")
    with pytest.raises(TypeError, match="generator"):
        s.minpoly(s.sqrt(2), s.Integer(3))
    with pytest.raises(PolynomialError, match="non-commutative"):
        s.minpoly(s.sqrt(2), s.Symbol("N", commutative=False))
