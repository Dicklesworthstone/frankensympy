"""Exact affine families and safe nonlinear dispatch through the real bridge."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def family(equations, *variables):
    result = s.nonlinsolve(equations, *variables)
    assert len(result) == 1
    return tuple(next(iter(result)))


def test_underdetermined_redundant_system_keeps_every_free_direction():
    x, y, z = s.symbols("affine_nls_x affine_nls_y affine_nls_z")
    equations = [3*x + 6*y - 3*z - 9, s.Eq(2*x + 4*y - 2*z, 6, evaluate=False)]
    answer = family(equations, x, y, z)
    assert answer == (3 - 2*y + z, y, z)
    assert answer[1] is y
    assert answer[2] is z
    for second, third in [(0, 0), (2, -1), (s.Rational(2, 3), s.Rational(-4, 5))]:
        first = 3 - 2*second + third
        assert 3*first + 6*second - 3*third - 9 == 0
        assert 2*first + 4*second - 2*third == 6


def test_rational_overdetermined_unique_solution_and_explicit_generator_order():
    x, y = s.symbols("affine_unique_x affine_unique_y")
    equations = [3*x - 1, 5*y + 2, 15*x + 15*y + 1]
    assert family(equations, x, y) == (s.Rational(1, 3), s.Rational(-2, 5))
    assert family(equations, [y, x]) == (s.Rational(-2, 5), s.Rational(1, 3))
    assert family([2*x + y - 5], y, x) == (5 - 2*x, x)


def test_present_zero_equations_preserve_all_supplied_variables():
    x, y = s.Dummy("zero_coordinate"), s.Symbol("zero_coordinate", real=True)
    answer = family([0, s.Eq(x, x, evaluate=False)], x, y)
    assert answer[0] is x
    assert answer[1] is y
    assert s.nonlinsolve([], x, y) == s.EmptySet()


def test_inconsistent_affine_systems_are_exact_empty_sets():
    x, y = s.symbols("affine_empty_x affine_empty_y")
    epsilon = s.Rational(1, 10**80)
    assert s.nonlinsolve([x + 2*y - 1, 2*x + 4*y - 2 - epsilon], x, y) == s.EmptySet()
    assert s.nonlinsolve([0, 7], x, y) == s.EmptySet()


def test_tiny_nonzero_pivot_is_not_mistaken_for_a_free_direction():
    x, y = s.symbols("affine_tiny_x affine_tiny_y")
    epsilon = s.Rational(1, 10**80)
    equations = [x + y - 3, x + (1 + epsilon)*y - 3 - 2*epsilon]
    assert family(equations, x, y) == (1, 2)


def test_same_spelling_dummy_and_assumed_parameters_keep_identity():
    first, second = s.Dummy("shared_coordinate"), s.Dummy("shared_coordinate")
    answer = family([first + second - 2], first, second)
    assert answer == (2 - second, second)
    assert answer[1] is second
    assert answer[0].free_symbols == {second}
    plain = s.Symbol("shared_coordinate")
    assumed = s.Symbol("shared_coordinate", positive=True)
    typed = family([plain + assumed - 3], plain, assumed)
    assert typed == (3 - assumed, assumed)
    assert typed[1] is assumed
    assert typed[0].free_symbols == {assumed}


def test_inferred_order_distinguishes_same_name_assumptions_and_dummies():
    variables = [
        s.Symbol("inferred_coordinate"),
        s.Symbol("inferred_coordinate", positive=True),
        s.Symbol("inferred_coordinate", integer=True),
        s.Dummy("inferred_coordinate"),
        s.Dummy("inferred_coordinate"),
    ]
    values = {variable: s.Integer(index + 1) for index, variable in enumerate(variables)}

    def declared_order(variable):
        binding = _symbol_binding(variable)
        return variable.name, binding.name, binding.identity_hex

    expected = tuple(values[variable] for variable in sorted(variables, key=declared_order))
    equations = [variable - values[variable] for variable in variables]
    for sources in [equations, list(reversed(equations)), equations[2:] + equations[:2]]:
        assert family(sources) == expected
        assert s.solve_poly_system(sources) == [expected]


def test_iterables_are_consumed_once_for_affine_and_nonlinear_systems():
    x, y = s.symbols("affine_iter_x affine_iter_y")
    assert family((equation for equation in [x + y - 3]), x, y) == (3 - y, y)
    result = s.nonlinsolve((equation for equation in [x*x - 4, y - x]), x, y)
    assert result == s.FiniteSet((-2, -2), (2, 2))


@pytest.mark.parametrize("power", [-1, 0])
def test_undefined_originals_are_rejected_before_affine_or_nonlinear_dispatch(power):
    x, y = s.symbols("affine_held_x affine_held_y")
    undefined = s.Pow(0, power, evaluate=False)
    hidden = s.Add(y, s.Mul(0, undefined, evaluate=False), evaluate=False)
    for first in [x - 1, x*x - 1, s.Integer(1)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.nonlinsolve([first, hidden], x, y)
    inverse = s.Pow(x, -1, evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.nonlinsolve([s.Eq(inverse, inverse, evaluate=False), x - 1], x)


def test_nonlinear_dispatch_retains_complete_finite_results_and_explicit_refusals():
    x, y = s.symbols("affine_fallback_x affine_fallback_y")
    assert s.nonlinsolve([x*x - 4, y*y - 9], x, y) == s.FiniteSet((-2, -3), (-2, 3), (2, -3), (2, 3))
    assert s.nonlinsolve([x*x - 4, x - 1], x) == s.EmptySet()
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([x*y], x, y)
    with pytest.raises((ValueError, NotImplementedError)):
        s.nonlinsolve([x**5 - x + 1, y - x], x, y)


def test_native_bridge_returns_exact_rows_only_after_admitting_every_original():
    x, y = s.symbols("affine_bridge_x affine_bridge_y")
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    rows = _native.polynomial_affine_matrix_native(
        [_native_expr(2*x/3 - 4*y/5 + s.Rational(7, 11)), _native_expr(0)], bindings
    )
    assert s.Matrix(rows) == s.Matrix([[s.Rational(2, 3), s.Rational(-4, 5), s.Rational(-7, 11)], [0, 0, 0]])
    assert _native.polynomial_affine_matrix_native([_native_expr(x*y)], bindings) is None
    hidden = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        _native.polynomial_affine_matrix_native([_native_expr(x*y), _native_expr(hidden)], bindings)


def test_nonpolynomial_sources_and_generator_limits_remain_explicit():
    x, y = s.symbols("affine_refusal_x affine_refusal_y")
    hidden_float = s.Add(x, s.Mul(0, s.Float(0.125), evaluate=False), evaluate=False)
    for source in [hidden_float, s.sin(x), s.Pow(x, 257, evaluate=False), y - 1]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.nonlinsolve([source], x)
    with pytest.raises(ValueError):
        s.nonlinsolve([x]*65, x)
    with pytest.raises(ValueError):
        s.nonlinsolve([0], *s.symbols("affine_boundary0:9"))
    with pytest.raises(ValueError):
        s.nonlinsolve([0], x, x)
    with pytest.raises(ValueError):
        _native.polynomial_affine_matrix_native([], [_symbol_binding(x), _symbol_binding(x)])
