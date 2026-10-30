"""Complete unions of exact polynomial families through public nonlinsolve."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def solutions(equations, *variables):
    return {tuple(row) for row in s.nonlinsolve(equations, *variables)}


def test_coordinate_axes_are_both_returned():
    x, y = s.symbols("components_x components_y")
    assert solutions([x*y], x, y) == {(0, y), (x, 0)}
    assert solutions([x*y], y, x) == {(y, 0), (0, x)}


def test_monomial_powers_retain_all_distinct_coordinate_planes():
    x, y, z = s.symbols("planes_x planes_y planes_z")
    expected = {(0, y, z), (x, 0, z), (x, y, 0)}
    assert solutions([x*y*z], x, y, z) == expected
    assert solutions([s.Rational(-7, 13)*x**4*y**2*z**3], x, y, z) == expected


def test_expanded_and_factored_parabola_unions_agree():
    x, y = s.symbols("parabolas_x parabolas_y")
    expected = {(y**2, y), (-y**2, y)}
    for equation in [x*x-y**4, (x-y*y)*(x+y*y)]:
        assert solutions([equation], x, y) == expected


def test_rational_scaled_difference_of_squares_keeps_both_components():
    x, y = s.symbols("scaled_components_x scaled_components_y")
    assert solutions([4*x*x-9*y**4], x, y) == {
        (s.Rational(3, 2)*y*y, y),
        (-s.Rational(3, 2)*y*y, y),
    }


def test_component_chains_preserve_caller_order_and_free_parameters():
    x, y, z = s.symbols("component_chain_x component_chain_y component_chain_z")
    equations = [(x-y*y)*(x+y*y), z-x**3]
    expected = {(y*y, y, y**6), (-y*y, y, -y**6)}
    assert solutions(equations, x, y, z) == expected
    assert solutions(list(reversed(equations)), z, x, y) == {
        (y**6, y*y, y), (-y**6, -y*y, y),
    }


def test_intersecting_unions_produce_every_rational_point():
    x, y = s.symbols("component_points_x component_points_y")
    equations = [x*y, (x-1)*(y-1)]
    expected = {(0, 1), (1, 0)}
    assert solutions(equations, x, y) == expected
    assert set(s.solve_poly_system(equations, x, y)) == expected


def test_subfamilies_are_removed_only_when_containment_is_established():
    x, y = s.symbols("component_containment_x component_containment_y")
    # The y=0 split yields the origin, already covered by the x=0 family.
    assert solutions([x*y, x*(y-1)], x, y) == {(0, y)}
    assert solutions([x*y, y*(x-1)], x, y) == {(x, 0)}


def test_redundant_constraints_and_repeated_factors_do_not_duplicate_families():
    x, y = s.symbols("repeated_components_x repeated_components_y")
    relation = (x-y*y)*(x+y*y)
    assert solutions([relation, 3*relation], x, y) == {(y*y, y), (-y*y, y)}
    assert solutions([(x-y*y)**4], x, y) == {(y*y, y)}


def test_same_name_symbol_and_dummy_parameters_keep_their_identity():
    plain = s.Symbol("component_parameter")
    assumed = s.Symbol("component_parameter", real=True)
    result = solutions([plain*assumed], plain, assumed)
    assert result == {(0, assumed), (plain, 0)}
    nonzero_symbols = [entry for row in result for entry in row if entry != 0]
    assert {id(value) for value in nonzero_symbols} == {id(plain), id(assumed)}
    first, second = s.Dummy("component_parameter"), s.Dummy("component_parameter")
    result = solutions([first*second], first, second)
    assert result == {(0, second), (first, 0)}
    assert {id(entry) for row in result for entry in row if entry != 0} == {id(first), id(second)}


def test_one_unsupported_component_prevents_a_partial_answer():
    x, y = s.symbols("partial_components_x partial_components_y")
    # y=0 is a polynomial graph, but the two irrational x=+/-sqrt(2)
    # families require algebraic coefficients beyond this component lane.
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([y*(x*x-2)], x, y)
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([x*(x*y-1)], x, y)


def test_finite_solver_does_not_return_positive_dimensional_families():
    x, y = s.symbols("finite_components_x finite_components_y")
    with pytest.raises(NotImplementedError):
        s.solve_poly_system([x*y], x, y)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_every_original_is_checked_before_any_component_can_be_returned(exponent):
    x, y = s.symbols("guarded_components_x guarded_components_y")
    hidden = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    for equations in [[x*y, hidden], [s.Integer(1), x*y, hidden]]:
        with pytest.raises(ValueError):
            s.nonlinsolve(equations, x, y)


def test_single_pass_inputs_and_exact_empty_unions():
    x, y = s.symbols("iter_components_x iter_components_y")
    assert solutions((equation for equation in [x*y, (x-1)*(y-1)]), x, y) == {(0, 1), (1, 0)}
    assert s.nonlinsolve([x*y, x*y-1], x, y) == s.EmptySet()


def test_native_dispatch_publishes_all_verified_family_rows():
    x, y = s.symbols("native_components_x native_components_y")
    result = _native.polynomial_system_solutions_native(
        [_native_expr(x*y)], [_symbol_binding(x), _symbol_binding(y)], True,
    )
    assert result.kind == "family"
    assert len(result.solutions) == 2
    assert all(len(row) == 2 for row in result.solutions)
