"""Rational fibers of finite rational ideals through the public solver."""

import pytest

from sympy import (
    CRootOf, Dummy, Eq, Mul, Poly, Pow, Rational, Symbol, solve_triangulated, symbols,
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


def test_algebraic_extension_selects_paired_conjugates_in_distinct_coordinates():
    x, y = symbols("x y")
    a = [CRootOf(x**2-2*x-1, i) for i in (0, 1)]
    b = [CRootOf(x**2-2, i) for i in (0, 1)]
    assert solve_triangulated([x-y-1, (y**2-2)**2], x, y, extension=True) == list(zip(a, b))
    assert solve_triangulated([x-y-1, (y**2-2)**2], y, x, extension=True) == list(zip(b, a))


def test_independent_quadratic_extensions_keep_all_four_embeddings():
    x, y = symbols("x y")
    a = [CRootOf(x**2-2, i) for i in (0, 1)]
    b = [CRootOf(x**2-3, i) for i in (0, 1)]
    assert solve_triangulated([x**2-2, y**2-3], x, y, extension=True) == [
        (u, v) for u in a for v in b]


def test_extension_keeps_rational_points_and_shared_real_embeddings():
    x, y, z = symbols("x y z")
    f = [x*x+y+z-1, x+y*y+z-1, x+y+z*z-1]
    r = [CRootOf(x*x+2*x-1, i) for i in (0, 1)]
    assert solve_triangulated(f, x, y, z, extension=True) == [
        (0, 0, 1), (0, 1, 0), (1, 0, 0), (r[0],)*3, (r[1],)*3]


def test_nonreal_projection_refuses_even_if_other_points_are_rational_or_real():
    x, y = symbols("x y")
    with pytest.raises(NotImplementedError, match="totally real"):
        solve_triangulated([x-y, (y-1)*(y**2+1)], x, y, extension=True)


def test_variable_order_uses_existing_univariate_constraint_and_restores_caller_order():
    x, y = symbols("x y")
    f = [(3*y-1)*(y+3)*(y**2-2), (x-y**2)**2]
    assert solve_triangulated(f, x, y) == [(Rational(1, 9), Rational(1, 3)), (9, -3)]
    assert solve_triangulated(f[::-1], y, x) == [(-3, 9), (Rational(1, 3), Rational(1, 9))]


def test_zero_leading_coefficient_in_a_source_fiber_cannot_remove_points():
    x, y = symbols("x y")
    f = [y*(y-1), y*x, x*(x-2)]
    assert solve_triangulated(f, x, y) == [(0, 0), (0, 1), (2, 0)]
    assert solve_triangulated(f, y, x) == [(0, 0), (0, 2), (1, 0)]


def test_irreducible_quintic_keeps_every_complex_embedding_once(monkeypatch):
    from sympy.polys import rootoftools

    x = Symbol("x")
    f = x**5-x+1
    expected = [(CRootOf(f, i),) for i in range(5)]

    def no_approximations(*args):
        raise AssertionError("complete local index enumeration must not select roots numerically")

    monkeypatch.setattr(rootoftools, "_approximate_factor_roots", no_approximations)
    roots = solve_triangulated([f**2], x, extension=True)
    assert roots == expected
    assert [root.poly.as_expr() for root, in roots] == [f]*5


def test_joint_source_equations_reduce_to_independent_complex_axes():
    x, y = symbols("x y")
    f, g = x**2+1, y**2-2
    # Neither original equation involves only one axis.
    equations = [f+g, f-g]
    expected = [(CRootOf(x**2+1, i), CRootOf(x**2-2, j))
                for i in range(2) for j in range(2)]
    assert solve_triangulated(equations, x, y, extension=True) == expected


def test_complex_and_rational_factors_preserve_indices_and_remove_multiplicities():
    x, y = symbols("x y")
    f = [(x-1)*(x**2+1)**2, y**2+2]
    a = [1, CRootOf(x**2+1, 0), CRootOf(x**2+1, 1)]
    b = [CRootOf(x**2+2, 0), CRootOf(x**2+2, 1)]
    assert solve_triangulated(f, x, y, extension=True) == [(u, v) for u in a for v in b]


def test_positive_integer_scaling_preserves_canonical_complex_indices_and_real_pairs():
    x, y = symbols("x y")
    f = x**5+512*x+1024
    assert solve_triangulated([f], x, extension=True) == [
        (4*CRootOf(x**5+2*x+1, i),) for i in range(5)]
    roots = [2*CRootOf(x**2-2*x-1, i) for i in (0, 1)]
    assert solve_triangulated([x**2-4*x-4, y-x], x, y, extension=True) == [
        (root, root) for root in roots]
