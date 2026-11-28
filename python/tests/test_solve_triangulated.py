"""Rational fibers of finite rational ideals through the public solver."""

import pytest

from sympy import (
    Dummy, Eq, Mul, Poly, Pow, Rational, Symbol, solve_triangulated, symbols,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers import solve_triangulated as solver_export


def test_gianni_kalkbrenner_example_separates_ground_and_extension_points():
    x, y, z = symbols("x y z")
    source = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]
    expected = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    assert solver_export(source, x, y, z) == expected
    for point in expected:
        assert all(eq.subs(dict(zip((x, y, z), point))) == 0 for eq in source)


def test_degree_dropping_fiber_keeps_all_and_only_compatible_extensions():
    x, y = symbols("x y")
    source = [x*(x-1), y*(y-1), x*y]
    assert solve_triangulated(source, x, y) == [(0, 0), (0, 1), (1, 0)]
    assert solve_triangulated(source[::-1] + [x*y], y, x) == [(0, 0), (0, 1), (1, 0)]


def test_rational_branch_of_a_system_with_an_irreducible_quintic():
    x = Dummy("v")
    y = Symbol("v", real=True)
    root = Rational(-1, 3)
    source = [Eq(x, 2*y-1, evaluate=False), Poly((3*y+1)**2*(y**5-y-1), y)]
    assert solve_triangulated(source, x, y) == [(2*root-1, root)]
    assert solve_triangulated(source, y, x) == [(root, 2*root-1)]


def test_unsatisfiable_and_algebraic_only_systems_both_have_no_ground_points():
    x, y = symbols("x y")
    assert solve_triangulated([x-1, x-2, y], x, y) == []
    assert solve_triangulated([x**2-2, x-y], x, y) == []


def test_integral_coefficients_may_have_fractional_ground_roots():
    x = Symbol("x")
    assert solve_triangulated([(2*x+3)**2*(5*x-1)], x, domain="ZZ") == [
        (Rational(-3, 2),), (Rational(1, 5),)]
    assert solve_triangulated([2*x-1]) == [(Rational(1, 2),)]


def test_refusals_do_not_publish_an_empty_or_partial_set():
    x, y = symbols("x y")
    with pytest.raises(PolynomialError, match="zero-dimensional"):
        solve_triangulated([x*y], x, y)
    with pytest.raises(ValueError):
        solve_triangulated([1, Mul(0, Pow(x, -1, evaluate=False), evaluate=False)], x)
    with pytest.raises(NotImplementedError):
        solve_triangulated([Poly(x+1, x, modulus=3)], x)
    with pytest.raises(NotImplementedError):
        solve_triangulated([x**2+1], x, extension=True)


def test_branch_limit_never_publishes_a_truncated_product():
    x, y, z = symbols("x y z")
    equations = []
    for variable in (x, y, z):
        equation = 1
        for value in range(7):
            equation *= variable - value
        equations.append(equation)
    with pytest.raises(ValueError, match="branch limit"):
        solve_triangulated(equations, x, y, z)
