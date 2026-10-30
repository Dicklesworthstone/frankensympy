"""Polynomial graph families and complete all-real algebraic finite systems.

These tests exercise the actual native dispatcher; upstream SymPy is not a
production fallback or an alternative execution harness for this file.
"""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def family(equations, *variables):
    result = s.nonlinsolve(equations, *variables)
    assert len(result) == 1
    return tuple(next(iter(result)))


def quintic(variable):
    # Sign changes at -2, -3/2, -1, 0, 1, 2 establish five distinct real
    # roots independently of any binary64 approximation.
    return variable**5 + variable**4 - 4*variable**3 - 3*variable**2 + 3*variable + 1


def dispatch(equations, variables, *, allow_families=False):
    return _native.polynomial_system_solutions_native(
        [_native_expr(equation) for equation in equations],
        [_symbol_binding(variable) for variable in variables],
        allow_families,
    )


def test_parabola_families_work_in_both_orientations_and_caller_orders():
    x, y = s.symbols("graph_x graph_y")
    assert family([x-y**2], x, y) == (y**2, y)
    assert family([y-x**2], x, y) == (x, x**2)
    assert family([x-y**2], y, x) == (y, y**2)
    with pytest.raises(NotImplementedError):
        s.solve_poly_system([x-y**2], x, y)


def test_polynomial_surfaces_retain_every_free_parameter():
    x, y, z, w = s.symbols("surface_x surface_y surface_z surface_w")
    answer = family([3*z-2*x*y-1], x, y, z, w)
    assert answer == (x, y, s.Rational(2, 3)*x*y + s.Rational(1, 3), w)
    assert answer[0] is x
    assert answer[1] is y
    assert answer[3] is w
    assert answer[2].free_symbols == {x, y}


def test_graph_chains_and_redundant_equations_eliminate_all_pivot_parameters():
    x, y, z = s.symbols("chain_x chain_y chain_z")
    equations = [x-y**2, 3*z-2*x*y-1, (x-y**2)*(z+1)]
    answer = family(equations, x, y, z)
    assert answer == (y**2, y, s.Rational(2, 3)*y**3 + s.Rational(1, 3))
    assert set().union(*(coordinate.free_symbols for coordinate in answer)) == {y}
    assert family(list(reversed(equations)), z, x, y) == (answer[2], answer[0], y)


def test_nonlinear_generators_can_jointly_define_one_complete_graph():
    x, y, z = s.symbols("hidden_graph_x hidden_graph_y hidden_graph_z")
    equations = [(x-y**2)*(z**2+1), (x-y**2)*(z**2+2)]
    assert family(equations, x, y, z) == (y**2, y, z)


def test_typed_same_name_and_dummy_free_coordinates_keep_identity():
    plain = s.Symbol("graph_coordinate")
    assumed = s.Symbol("graph_coordinate", positive=True)
    answer = family([plain-assumed**2], plain, assumed)
    assert answer == (assumed**2, assumed)
    assert answer[1] is assumed
    assert answer[0].free_symbols == {assumed}
    first, second = s.Dummy("graph_coordinate"), s.Dummy("graph_coordinate")
    answer = family([first-second**3], first, second)
    assert answer == (second**3, second)
    assert answer[1] is second
    assert answer[0].free_symbols == {second}


def test_rational_graphs_are_supported_and_curved_multivalued_families_refuse():
    x, y = s.symbols("nongraph_x nongraph_y")
    assert family([x*y-1], x, y) == (1/y, y)
    with pytest.raises(NotImplementedError):
        s.nonlinsolve([x*x+y*y-1], x, y)


def test_affine_dispatch_keeps_zero_system_and_empty_input_conventions():
    x, y = s.symbols("affine_dispatch_x affine_dispatch_y")
    assert family([0, s.Eq(x, x, evaluate=False)], x, y) == (x, y)
    assert s.nonlinsolve([], x, y) == s.EmptySet()
    assert family([3*x-1, 5*y+2], x, y) == (s.Rational(1, 3), s.Rational(-2, 5))
    assert s.nonlinsolve([x-y*y, x-y*y-1], x, y) == s.EmptySet()


def test_all_real_irreducible_quintic_produces_every_exact_shape_tuple():
    x, z, y = s.symbols("shape_x shape_z shape_y")
    polynomial = quintic(y)
    result = s.solve_poly_system([x-y**2-1, z-2*y+1, polynomial], x, z, y)
    assert len(result) == 5
    expected_roots = [s.CRootOf(polynomial, y, index) for index in range(5)]
    assert [solution[2] for solution in result] == expected_roots
    for solution, root in zip(result, expected_roots):
        assert solution == (1 + root**2, -1 + 2*root, root)
        assert root.is_real is True
        assert all(not coordinate.free_symbols for coordinate in solution)
    assert len(set(result)) == 5


def test_shape_results_are_exact_for_single_variable_and_nonlinsolve_sets():
    y = s.Symbol("single_shape_y")
    polynomial = quintic(y)
    expected = [s.CRootOf(polynomial, y, index) for index in range(5)]
    assert s.solve_poly_system([polynomial], y) == [(root,) for root in expected]
    result = s.nonlinsolve([polynomial], y)
    assert len(result) == 5
    assert {tuple(solution) for solution in result} == {(root,) for root in expected}


def test_shape_multiplicity_does_not_duplicate_distinct_solution_tuples():
    x, y = s.symbols("multiple_shape_x multiple_shape_y")
    polynomial = quintic(y)
    result = s.solve_poly_system([x-y, polynomial**2], x, y)
    assert len(result) == 5
    assert all(first == second for first, second in result)
    native = dispatch([x-y, polynomial**2], [x, y])
    assert native.kind == "real_shape"
    assert list(native.multiplicities) == [2]*5


def test_shape_lifting_keeps_rational_roots_and_factor_multiplicities_together():
    x, y = s.symbols("mixed_factor_shape_x mixed_factor_shape_y")
    irreducible = quintic(y)
    polynomial = (y-3)*irreducible**2
    result = s.solve_poly_system([2*x-y-1, polynomial], x, y)
    roots = [s.CRootOf(irreducible, y, index) for index in range(5)] + [s.Integer(3)]
    assert result == [(s.Rational(1, 2)*root + s.Rational(1, 2), root) for root in roots]
    assert result[-1] == (2, 3)
    native = dispatch([2*x-y-1, polynomial], [x, y])
    assert native.kind == "real_shape"
    assert list(native.multiplicities) == [2]*5 + [1]


def test_complex_domain_never_receives_only_the_real_subset_of_a_quintic():
    x, y = s.symbols("mixed_shape_x mixed_shape_y")
    equations = [x-y, y**5-y+1]
    for allow_families in [False, True]:
        with pytest.raises(ValueError):
            dispatch(equations, [x, y], allow_families=allow_families)
    with pytest.raises(ValueError):
        s.solve_poly_system(equations, x, y)
    with pytest.raises(ValueError):
        s.nonlinsolve(equations, x, y)


def test_existing_explicit_complex_solutions_remain_complete():
    x, y = s.symbols("complex_dispatch_x complex_dispatch_y")
    expected = {(-s.I, -s.I), (s.I, s.I)}
    assert set(s.solve_poly_system([x*x+1, y-x], x, y)) == expected
    assert {tuple(solution) for solution in s.nonlinsolve([x*x+1, y-x], x, y)} == expected
    assert dispatch([x*x+1, y-x], [x, y]).kind == "finite"


@pytest.mark.parametrize("power", [-1, 0])
def test_held_undefined_children_are_admitted_before_every_dispatch_branch(power):
    x, y = s.symbols("source_dispatch_x source_dispatch_y")
    hidden = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
    for first in [s.Integer(1), x+y, x-y*y, quintic(y)]:
        for allow_families in [False, True]:
            with pytest.raises(ValueError):
                dispatch([first, hidden], [x, y], allow_families=allow_families)
        with pytest.raises(ValueError):
            s.nonlinsolve([first, hidden], x, y)
    inverse = s.Pow(x, -1, evaluate=False)
    with pytest.raises(ValueError):
        s.nonlinsolve([s.Eq(inverse, inverse, evaluate=False), x-y*y], x, y)


def test_source_and_generator_limits_apply_before_dispatch():
    x, y = s.symbols("bounded_dispatch_x bounded_dispatch_y")
    for equation in [s.Pow(x, 257, evaluate=False), s.Pow(2, 10**10, evaluate=False)]:
        with pytest.raises(ValueError):
            dispatch([equation], [x, y], allow_families=True)
    with pytest.raises(ValueError):
        dispatch([s.Integer(0)]*65, [x, y], allow_families=True)
    with pytest.raises(ValueError):
        dispatch([], [x, x], allow_families=True)
    with pytest.raises(ValueError):
        dispatch([], list(s.symbols("bounded_generator0:9")), allow_families=True)
    hidden_float = s.Add(x, s.Mul(0, s.Float(0.25), evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.nonlinsolve([hidden_float, x-y*y], x, y)


def test_empty_input_still_validates_the_complete_typed_generator_ring():
    oversized = s.Symbol("g"*65_537)
    for allow_families in [False, True]:
        with pytest.raises(ValueError, match="generator boundary exceeded"):
            dispatch([], [oversized], allow_families=allow_families)
    first = s.Symbol("a"*32_769)
    second = s.Symbol("b"*32_769)
    with pytest.raises(ValueError, match="generator boundary exceeded"):
        dispatch([], [first, second], allow_families=True)


def test_single_pass_iterators_reach_graph_and_shape_dispatch():
    x, y = s.symbols("iterator_dispatch_x iterator_dispatch_y")
    assert family((equation for equation in [x-y*y]), x, y) == (y*y, y)
    result = s.solve_poly_system((equation for equation in [x-y, quintic(y)]), x, y)
    assert len(result) == 5
    assert all(first == second for first, second in result)
