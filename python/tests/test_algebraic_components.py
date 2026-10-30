"""Complete finite algebraic loci retain the caller's free coordinates."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def points(equations, *variables):
    return {tuple(row) for row in s.nonlinsolve(equations, *variables)}


def dispatch(equations, *variables):
    return _native.polynomial_system_solutions_native(
        [_native_expr(equation) for equation in equations],
        [_symbol_binding(variable) for variable in variables],
        True,
    )


def test_both_real_quadratic_conjugates_keep_the_free_coordinate():
    x, y = s.symbols("algebraic_cylinder_x algebraic_cylinder_y")
    assert points([x*x-2], x, y) == {(-s.sqrt(2), y), (s.sqrt(2), y)}
    result = dispatch([x*x-2], x, y)
    assert result.kind == "family" and len(result.solutions) == 2
    assert all(len(row) == 2 for row in result.solutions)


def test_complex_roots_and_two_free_coordinates_keep_requested_order():
    x, y, z = s.symbols("algebraic_complex_x algebraic_complex_y algebraic_complex_z")
    assert points([x*x+1], y, x, z) == {(y, -s.I, z), (y, s.I, z)}
    assert all(row[0] is y and row[2] is z for row in points([x*x+1], y, x, z))


def test_coupled_active_equations_preserve_all_coordinates_and_constraints():
    x, y, z, w = s.symbols("algebraic_coupled_x algebraic_coupled_y algebraic_coupled_z algebraic_coupled_w")
    equations = [x*x-2, z-x, 2*z-2*x, 0]
    assert points((equation for equation in equations), (z, y, x, w)) == {
        (-s.sqrt(2), y, -s.sqrt(2), w), (s.sqrt(2), y, s.sqrt(2), w),
    }


def test_independent_quadratics_return_every_conjugate_combination():
    x, y, z = s.symbols("algebraic_cartesian_x algebraic_cartesian_y algebraic_cartesian_z")
    expected = {(first, y, second) for first in [-s.sqrt(3), s.sqrt(3)] for second in [-s.sqrt(2), s.sqrt(2)]}
    assert points([x*x-2, z*z-3], z, y, x) == expected


@pytest.mark.parametrize("repeated", [False, True])
def test_reducible_union_keeps_the_entire_free_line_and_both_algebraic_lines(repeated):
    x, y = s.symbols("algebraic_union_x algebraic_union_y")
    source = y*y*(x*x-2)**3 if repeated else y*(x*x-2)
    assert points([source], x, y) == {(x, 0), (-s.sqrt(2), y), (s.sqrt(2), y)}
    assert points([source], y, x) == {(0, x), (y, -s.sqrt(2)), (y, s.sqrt(2))}


def test_mixed_rational_and_algebraic_components_keep_separate_domains():
    x, y = s.symbols("algebraic_rational_x algebraic_rational_y")
    result = points([(x*x-2)*(x*y-1)], x, y)
    assert result == {(-s.sqrt(2), y), (s.sqrt(2), y), (1/y, y)}
    for row in result:
        assert row[1] is y
    metadata = dispatch([(x*x-2)*(x*y-1)], x, y)
    assert metadata.kind == "rational_families"
    assert sorted(len(guards) for guards in metadata.family_nonzero_conditions) == [0, 0, 1]


def test_isolated_algebraic_points_are_not_lost_beside_a_free_family():
    # Exact source constraints require all three components. This is a
    # mathematical completeness regression, not a pinned-oracle parity claim.
    x, y = s.symbols("algebraic_isolated_x algebraic_isolated_y")
    assert points([x*(x*x-2), x*y], x, y) == {
        (0, y), (-s.sqrt(2), 0), (s.sqrt(2), 0),
    }


def test_same_printing_symbol_and_dummy_objects_keep_their_original_roles():
    plain = s.Symbol("algebraic_identity")
    assumed = s.Symbol("algebraic_identity", real=True)
    result = points([assumed*assumed-2], plain, assumed)
    assert result == {(plain, -s.sqrt(2)), (plain, s.sqrt(2))}
    assert all(row[0] is plain for row in result)
    first, second = s.Dummy("algebraic_dummy"), s.Dummy("algebraic_dummy")
    result = points([first*first+1], second, first)
    assert result == {(second, -s.I), (second, s.I)}
    assert all(row[0] is second for row in result)


def test_an_unsolved_component_prevents_a_partial_public_solution_set():
    x, y = s.symbols("algebraic_unsolved_x algebraic_unsolved_y")
    for source in [y*(x**5-x+1), (x*x-2)*(x*x+y*y-1), x*x+y*y-1]:
        with pytest.raises(NotImplementedError):
            s.nonlinsolve([source], x, y)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_every_original_domain_is_checked_before_a_cylinder_or_empty_answer(exponent):
    x, y = s.symbols("algebraic_domain_x algebraic_domain_y")
    bad = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    for equations in [[x*x-2, bad], [1, x*x-2, bad]]:
        with pytest.raises(ValueError):
            s.nonlinsolve(equations, x, y)


def test_finite_solver_keeps_its_finite_contract():
    x, y = s.symbols("algebraic_finite_x algebraic_finite_y")
    with pytest.raises(NotImplementedError):
        s.solve_poly_system([x*x-2], x, y)
    assert set(s.solve_poly_system([x*x-2, y], x, y)) == {(-s.sqrt(2), 0), (s.sqrt(2), 0)}


def test_native_operational_failure_does_not_return_a_partial_component(monkeypatch):
    x, y = s.symbols("algebraic_failure_x algebraic_failure_y")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("algebraic component verifier budget exhausted")

    monkeypatch.setattr(_native, "polynomial_system_solutions_native", exhausted)
    with pytest.raises(ValueError, match="verifier budget exhausted"):
        s.nonlinsolve([y*(x*x-2)], x, y)
    assert len(calls) == 1
