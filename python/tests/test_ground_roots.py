"""Ground-field root sets, multiplicities, residue lifts and source admission."""

import pytest

from sympy import Dummy, Mul, Poly, Pow, PurePoly, Rational, Symbol, ground_roots
from sympy.polys import ground_roots as polynomial_ground_roots
from sympy.polys.polyerrors import MultivariatePolynomialError


@pytest.mark.parametrize("cls", [Poly, PurePoly])
def test_rational_grid_excludes_irrationals_and_keeps_multiplicities(cls):
    x = Dummy("x")
    source = ((5*x + 2)**3 * (7*x - 3)**2 * x * (x**2 - 2)
              * (x**2 + 1)) / 30
    expected = {Rational(-2, 5): 3, Rational(3, 7): 2, 0: 1}
    assert cls(source, x).ground_roots() == expected
    assert ground_roots(source) == expected
    assert polynomial_ground_roots(source, x, domain="QQ") == expected
    for root, multiplicity in expected.items():
        polynomial = cls(source, x).as_expr()
        for _ in range(multiplicity):
            assert polynomial.subs(x, root) == 0
            polynomial = polynomial.diff(x)
        assert polynomial.subs(x, root) != 0


def test_irrational_root_inside_a_rational_candidates_isolator_is_excluded():
    x = Symbol("x")
    assert ground_roots(x**3 - x - 1) == {}
    assert ground_roots((17*x - 16) * (19*x - 18) * (x**5 - x - 1)) == {
        Rational(16, 17): 1, Rational(18, 19): 1,
    }


@pytest.mark.parametrize("value", [0, -9])
def test_constants_have_empty_ground_root_dictionaries(value):
    x = Symbol("x")
    assert Poly(value, x).ground_roots() == ground_roots(value, x) == {}


@pytest.mark.parametrize("symmetric,expected", [
    (True, {-2: 2, 2: 2}), (False, {-2: 2, -3: 2}),
])
def test_prime_field_uses_declared_characteristic_and_lift(symmetric, expected):
    x = Symbol("x")
    source = (x**2 + 1)**2
    assert ground_roots(source, x, modulus=5, symmetric=symmetric) == expected
    assert Poly(source, x, modulus=5, symmetric=symmetric).ground_roots() == expected
    assert ground_roots(source, x, modulus=2) == {-1: 4}
    assert ground_roots(source, x) == {}


def test_held_denominator_and_wrong_ring_are_rejected_before_simplification():
    x, y = Symbol("x"), Symbol("y")
    with pytest.raises(MultivariatePolynomialError):
        ground_roots(x + y)
    with pytest.raises(ValueError):
        ground_roots(Mul(0, Pow(x, -1, evaluate=False), evaluate=False), x)
    with pytest.raises((ValueError, ArithmeticError)):
        ground_roots(x + y, x)
