"""Finite systems whose complete solution set needs a separating variable."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding, _wrap


def quintic(value):
    return value**5+value**4-4*value**3-3*value**2+3*value+1


def dispatch(equations, variables, families=False):
    return _native.polynomial_system_solutions_native(
        [_native_expr(value) for value in equations],
        [_symbol_binding(value) for value in variables],
        families,
    )


def test_separating_form_returns_all_ten_points_of_a_quintic_product():
    x, y = s.symbols("rur_x rur_y")
    equations = [quintic(x), y*y-2]
    solutions = s.solve_poly_system(equations, x, y)
    assert len(solutions) == len(set(solutions)) == 10
    for first, second in solutions:
        assert not first.free_symbols
        assert not second.free_symbols
        assert s.Eq(quintic(first), 0) is s.S.true
        assert s.Eq(second*second, 2) is s.S.true
    # Exact ordering resolves the two y embeddings after algebraic recovery.
    assert sum(value.is_positive is True for _, value in solutions) == 5
    assert sum(value.is_negative is True for _, value in solutions) == 5


def test_bridge_distinguishes_a_bound_separator_from_an_original_coordinate():
    x, y = s.symbols("rur_native_x rur_native_y")
    result = dispatch([quintic(x), y*y-2], [x, y])
    assert result.kind == "real_rur"
    assert len(result.eliminant) == 11
    assert len(result.coordinates) == 2
    assert result.multiplicities == [1]*10
    assert [_wrap(value) for value in result.coordinates[1]] != [s.Integer(0), s.Integer(1)]


def test_caller_order_is_preserved_after_inverse_coordinate_recovery():
    x, y = s.symbols("rur_order_x rur_order_y")
    solutions = s.solve_poly_system([quintic(x), y*y-2], y, x)
    assert len(solutions) == 10
    for quadratic_coordinate, quintic_coordinate in solutions:
        assert s.Eq(quadratic_coordinate**2, 2) is s.S.true
        assert s.Eq(quintic(quintic_coordinate), 0) is s.S.true


def test_nonlinsolve_and_single_pass_equations_publish_the_complete_set():
    x, y = s.symbols("rur_set_x rur_set_y")

    def equations():
        yield quintic(x)
        yield y*y-2

    result = s.nonlinsolve(equations(), (x, y))
    assert isinstance(result, s.FiniteSet)
    assert len(result) == 10
    assert all(len(point) == 2 and not point.free_symbols for point in result)


def test_source_symbol_identity_does_not_alias_the_bound_separator():
    x = s.Symbol("_fsym_rur_parameter", real=True)
    y = s.Dummy("_fsym_rur_parameter")
    result = dispatch([quintic(x), y*y-2], [x, y])
    assert result.kind == "real_rur"
    assert len(result.coordinates) == 2
    assert len(result.multiplicities) == 10
    # The public root binder can share a printed name without making it free.
    solutions = s.solve_poly_system([quintic(x), y*y-2], x, y)
    assert len(solutions) == 10
    assert all(not coordinate.free_symbols for point in solutions for coordinate in point)


def test_old_original_coordinate_shape_kind_is_preserved():
    x, y = s.symbols("rur_shape_x rur_shape_y")
    result = dispatch([x-y*y-1, quintic(y)], [x, y])
    assert result.kind == "real_shape"
    assert len(result.multiplicities) == 5
    assert [_wrap(value) for value in result.coordinates[1]] == [s.Integer(0), s.Integer(1)]


def test_nonreal_fibers_cannot_be_dropped_from_complex_domain_solutions():
    x, y = s.symbols("rur_complex_x rur_complex_y")
    for solve in (
        lambda equations: s.solve_poly_system(equations, x, y),
        lambda equations: s.nonlinsolve(equations, (x, y)),
    ):
        with pytest.raises(ValueError):
            solve([quintic(x), y*y+1])


@pytest.mark.parametrize("exponent", [-1, 0])
def test_all_original_held_children_are_checked_before_rur_dispatch(exponent):
    x, y = s.symbols("rur_original_x rur_original_y")
    bad = s.Pow(s.Integer(0), exponent, evaluate=False)
    hidden = s.Add(quintic(x), s.Mul(0, bad, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.solve_poly_system([hidden, y*y-2], x, y)


def test_recovered_coordinates_use_exact_algebraic_evalf():
    x, y = s.symbols("rur_eval_x rur_eval_y")
    solutions = s.solve_poly_system([quintic(x), y*y-2], x, y)
    for coordinate in solutions[0]:
        assert isinstance(s.N(coordinate, 50), s.Float)
        assert str(s.N(coordinate, 50)) == str(coordinate.evalf(50))
