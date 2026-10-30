"""Complete rational families retain parameter exclusions at the public edge."""

import pytest
import sympy as s
from sympy.core import _lift_native_result, _native, _native_expr, _symbol_binding


def family(equations, *variables):
    solutions = s.nonlinsolve(equations, *variables)
    assert len(solutions) == 1
    return tuple(next(iter(solutions)))


def dispatch(equations, variables, allow_families=True):
    return _native.polynomial_system_solutions_native(
        [_native_expr(equation) for equation in equations],
        [_symbol_binding(variable) for variable in variables],
        allow_families,
    )


def inverse_bases(expression):
    result = set()
    pending = [expression]
    while pending:
        node = pending.pop()
        if type(node) is s.Pow and node.args[1] == -1:
            result.add(node.args[0])
        pending.extend(node.args)
    return result


def test_hyperbola_is_complete_in_both_caller_orders():
    x, y = s.symbols("rational_family_x rational_family_y")
    forward = family([x*y-1], x, y)
    backward = family([x*y-1], y, x)
    assert forward == (1/y, y)
    assert backward == (1/x, x)
    assert forward[1] is y
    assert backward[1] is x
    assert inverse_bases(forward[0]) == {y}
    for parameter in [-5, -1, s.Rational(2, 3), 7]:
        assert forward[0].subs(y, parameter) * parameter == 1


def test_nonlinear_denominators_keep_their_complex_parameter_exclusions():
    x, y = s.symbols("rational_poles_x rational_poles_y")
    values = family([x*(y*y+1)-1], x, y)
    assert values[1] is y
    assert inverse_bases(values[0]) == {y*y+1}
    # The graph is over complex parameters, so +-I remain excluded even
    # though its denominator never vanishes for real y.
    denominator = next(iter(inverse_bases(values[0])))
    assert denominator.subs(y, s.I) == 0
    assert denominator.subs(y, -s.I) == 0
    for parameter in [0, 1, -2]:
        assert values[0].subs(y, parameter) * (parameter*parameter+1) == 1


def test_several_coordinates_share_one_guard_without_losing_any_coordinate():
    x, z, y = s.symbols("rational_shared_x rational_shared_z rational_shared_y")
    equations = [x*y-1, z*y-2]
    values = family(equations, x, z, y)
    assert values[2] is y
    assert inverse_bases(values[0]) == inverse_bases(values[1]) == {y}
    for parameter in [-3, 2, s.Rational(1, 5)]:
        assert values[0].subs(y, parameter) * parameter == 1
        assert values[1].subs(y, parameter) * parameter == 2
    result = dispatch(equations, [x, z, y])
    assert result.kind == "rational_family"
    assert len(result.numerators) == len(result.denominators) == 3
    assert len(result.nonzero_conditions) == 1


def test_coupled_rational_coordinates_can_be_found_from_the_checked_ideal():
    x, y, z = s.symbols("rational_hidden_x rational_hidden_y rational_hidden_z")
    relation = x*y-1
    equations = [relation*(z*z+1), relation*(z*z+2)]
    values = family(equations, x, y, z)
    assert values == (1/y, y, z)
    assert values[1] is y and values[2] is z


def test_free_symbols_with_identical_printing_keep_their_typed_identity():
    plain = s.Symbol("rational_identity")
    assumed = s.Symbol("rational_identity", real=True)
    values = family([plain*assumed-1], plain, assumed)
    assert values[1] is assumed
    assert values[0].free_symbols == {assumed}
    first, second = s.Dummy("rational_identity"), s.Dummy("rational_identity")
    values = family([first*second-1], first, second)
    assert values[1] is second
    assert values[0].free_symbols == {second}


def test_exceptional_fibers_prevent_publication_of_a_partial_rational_chart():
    x, y = s.symbols("rational_exceptional_x rational_exceptional_y")
    # x=1/y describes the hyperbola component but misses the whole x=0
    # component. A single rational graph cannot represent the complete set.
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([x*(x*y-1)], x, y)
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([x*x+y*y-1], x, y)


def test_polynomial_chart_can_avoid_a_division_with_exceptional_fibers():
    x, y, z = s.symbols("rational_alternate_x rational_alternate_y rational_alternate_z")
    values = family([x*y-z], x, y, z)
    assert values == (x, y, x*y)
    # In particular the y=0,z=0 fiber retains arbitrary x.
    assert values[2].subs(y, 0) == 0


def test_finite_systems_keep_the_existing_finite_result_contract():
    x, y = s.symbols("rational_finite_x rational_finite_y")
    equations = [x*y-1, y-2]
    assert family(equations, x, y) == (s.Rational(1, 2), 2)
    assert dispatch(equations, [x, y]).kind in ("family", "finite")
    assert s.solve_poly_system(equations, x, y) == [(s.Rational(1, 2), 2)]
    with pytest.raises(NotImplementedError):
        s.solve_poly_system([x*y-1], x, y)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_every_original_domain_is_admitted_before_empty_or_family_results(exponent):
    x, y = s.symbols("rational_original_x rational_original_y")
    invalid = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    for equations in [[x*y-1, invalid], [1, x*y-1, invalid]]:
        with pytest.raises(ValueError):
            s.nonlinsolve(equations, x, y)


def test_single_pass_inputs_and_native_guard_correspondence():
    x, y = s.symbols("rational_iter_x rational_iter_y")
    assert family((item for item in [x*y-1, 3*(x*y-1)]), x, y) == (1/y, y)
    result = dispatch([x*y-1], [x, y])
    assert result.kind == "rational_family"
    denominators = [_lift_native_result(value, x, y) for value in result.denominators]
    guards = [_lift_native_result(value, x, y) for value in result.nonzero_conditions]
    assert denominators == [y, 1]
    assert guards == [y]
    with pytest.raises(TypeError):
        _native.PolynomialSystemResult()
    with pytest.raises(AttributeError):
        result.kind = "finite"
