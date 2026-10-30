"""Complete unions keep each rational component's parameter domain separate."""

import pytest
import sympy as s
from sympy.core import _lift_native_result, _native, _native_expr, _symbol_binding


def solutions(equations, *variables):
    return {tuple(row) for row in s.nonlinsolve(equations, *variables)}


def dispatch(equations, *variables):
    return _native.polynomial_system_solutions_native(
        [_native_expr(equation) for equation in equations],
        [_symbol_binding(variable) for variable in variables],
        True,
    )


def inverse_bases(expression):
    bases = set()
    pending = [expression]
    while pending:
        node = pending.pop()
        if type(node) is s.Pow and node.args[1] == -1:
            bases.add(node.args[0])
        pending.extend(node.args)
    return bases


def test_complete_line_and_hyperbola_union_preserves_the_exceptional_fiber():
    x, y = s.symbols("mixed_x mixed_y")
    assert solutions([x*(x*y-1)], x, y) == {(0, y), (1/y, y)}
    reverse = solutions([x*(x*y-1)], y, x)
    assert reverse == {(y, 0), (1/x, x)}
    line = next(row for row in reverse if row[1] == 0)
    assert line[0] is y
    # The hyperbola excludes its parameter's zero; the line still contains
    # the origin and must not inherit the other component's exclusion.
    assert tuple(value.subs(y, 0) for value in line) == (0, 0)


def test_expanded_and_factored_unions_of_hyperbolas_are_complete():
    x, y = s.symbols("mixed_factored_x mixed_factored_y")
    expected = {(1/y, y), (-1/y, y)}
    for equation in [(x*y-1)*(x*y+1), x*x*y*y-1]:
        assert solutions([equation], x, y) == expected


def test_polynomial_and_rational_curves_have_different_parameter_domains():
    x, y = s.symbols("mixed_domains_x mixed_domains_y")
    result = solutions([(x-y*y)*(x*y-1)], x, y)
    assert result == {(y*y, y), (1/y, y)}
    polynomial = next(row for row in result if row[0] == y*y)
    assert inverse_bases(polynomial[0]) == set()
    assert polynomial[0].subs(y, 0) == 0


def test_complex_poles_are_local_to_the_rational_component():
    x, y = s.symbols("mixed_complex_x mixed_complex_y")
    result = solutions([x*(x*(y*y+1)-1)], x, y)
    assert len(result) == 2
    polynomial = next(row for row in result if row[0] == 0)
    rational = next(row for row in result if row[0] != 0)
    assert polynomial[1] is y and rational[1] is y
    assert inverse_bases(rational[0]) == {y*y+1}
    assert tuple(value.subs(y, s.I) for value in polynomial) == (0, s.I)
    assert (y*y+1).subs(y, s.I) == 0
    for value in [-2, 0, 3]:
        assert rational[0].subs(y, value) * (value*value+1) == 1


def test_two_rational_branches_do_not_inherit_each_others_poles():
    x, y = s.symbols("mixed_poles_x mixed_poles_y")
    result = solutions([(x*y-1)*(x*(y-1)-1)], x, y)
    assert len(result) == 2
    by_poles = {frozenset(inverse_bases(row[0])): row for row in result}
    assert set(by_poles) == {frozenset({y}), frozenset({y-1})}
    assert by_poles[frozenset({y})][0].subs(y, 1) == 1
    assert by_poles[frozenset({y-1})][0].subs(y, 0) == -1
    metadata = dispatch([(x*y-1)*(x*(y-1)-1)], x, y)
    assert metadata.kind == "rational_families"
    guards = {
        tuple(_lift_native_result(value, x, y) for value in row)
        for row in metadata.family_nonzero_conditions
    }
    assert guards == {(y,), (y-1,)}


def test_nonlinear_chains_retain_all_original_coordinates():
    x, z, y = s.symbols("mixed_chain_x mixed_chain_z mixed_chain_y")
    result = solutions([x*(x*y-1), z-x*x], x, z, y)
    assert len(result) == 2
    assert (0, 0, y) in result
    rational = next(row for row in result if row[0] != 0)
    assert rational[2] is y
    for value in [-3, s.Rational(2, 3), 4]:
        assert rational[0].subs(y, value) == s.Rational(1)/value
        assert rational[1].subs(y, value) == s.Rational(1)/(value*value)


def test_independent_mixed_choices_form_the_complete_cartesian_union():
    x, y, z, w = s.symbols("mixed_product_x mixed_product_y mixed_product_z mixed_product_w")
    assert solutions([x*(x*y-1), z*(z*w-1)], x, y, z, w) == {
        (0, y, 0, w), (0, y, 1/w, w),
        (1/y, y, 0, w), (1/y, y, 1/w, w),
    }


def test_repeated_factors_and_checked_containment_do_not_duplicate_components():
    x, y = s.symbols("mixed_repeat_x mixed_repeat_y")
    assert solutions([x*x*(x*y-1)**3], x, y) == {(0, y), (1/y, y)}
    # The x=1 branch yields the point (1,1), already on the hyperbola.
    equations = [(x*y-1)*(x-1), (x*y-1)*(y-1)]
    assert solutions(equations, x, y) == {(1/y, y)}
    assert dispatch(equations, x, y).kind == "rational_family"


def test_hidden_whole_ideal_chart_survives_unsupported_intermediate_splits():
    x, y, z = s.symbols("mixed_hidden_x mixed_hidden_y mixed_hidden_z")
    hyperbola = x*y-1
    assert solutions([hyperbola*(z*z+1), hyperbola*(z*z+2)], x, y, z) == {(1/y, y, z)}


def test_equal_printing_does_not_merge_typed_parameters_between_branches():
    plain = s.Symbol("mixed_identity")
    assumed = s.Symbol("mixed_identity", real=True)
    result = solutions([plain*(plain*assumed-1)], plain, assumed)
    assert result == {(0, assumed), (1/assumed, assumed)}
    assert all(row[1] is assumed for row in result)
    assert all(row[0].free_symbols <= {assumed} for row in result)


def test_original_dummy_objects_survive_free_coordinates_and_held_poles():
    first, second = s.Dummy("mixed_dummy"), s.Dummy("mixed_dummy")
    assert first != second
    assert _lift_native_result(_native_expr(first)) is first
    assert _lift_native_result(_native_expr(second)) is second
    equation = first*(first*second-1)
    result = solutions([equation], first, second)
    assert result == {(0, second), (1/second, second)}
    assert all(row[1] is second for row in result)
    rational = next(row for row in result if row[0] != 0)
    assert next(iter(rational[0].free_symbols)) is second
    assert next(iter(inverse_bases(rational[0]))) is second
    reversed_result = solutions([equation], second, first)
    assert reversed_result == {(second, 0), (1/first, first)}
    line = next(row for row in reversed_result if row[1] == 0)
    rational = next(row for row in reversed_result if row[1] != 0)
    assert line[0] is second and rational[1] is first
    assert next(iter(inverse_bases(rational[0]))) is first


def test_dummy_lifting_cannot_drop_an_unregistered_assumptions_identity():
    original = s.Dummy("mixed_dummy_identity")
    native_name = _symbol_binding(original).name
    distinct = _native.py_symbol(native_name, [("real", "True")])
    assert not distinct.symbol_binding.plain
    with pytest.raises(NotImplementedError, match="no registered surface object"):
        _lift_native_result(distinct, original)
    assert _lift_native_result(_native_expr(original)) is original


def test_unsolved_branch_still_prevents_any_partial_public_answer():
    x, y = s.symbols("mixed_unsupported_x mixed_unsupported_y")
    for equation in [x*(x*x+y*y-1), (x*y-1)*(x*x+y*y-1), y*(x*x-2)]:
        with pytest.raises(NotImplementedError):
            s.nonlinsolve([equation], x, y)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_all_original_domains_are_admitted_before_a_mixed_answer(exponent):
    x, y = s.symbols("mixed_held_x mixed_held_y")
    undefined = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    for equations in [[x*(x*y-1), undefined], [1, x*(x*y-1), undefined]]:
        with pytest.raises(ValueError):
            s.nonlinsolve(equations, x, y)


def test_finite_solver_keeps_its_finite_contract_and_single_pass_input():
    x, y = s.symbols("mixed_finite_x mixed_finite_y")
    equation = x*(x*y-1)
    assert solutions((item for item in [equation]), x, y) == {(0, y), (1/y, y)}
    assert set(s.solve_poly_system([equation, y-2], x, y)) == {(0, 2), (s.Rational(1, 2), 2)}
    with pytest.raises(NotImplementedError):
        s.solve_poly_system([equation], x, y)


def test_native_union_metadata_keeps_guards_per_component_and_is_immutable():
    x, y = s.symbols("mixed_metadata_x mixed_metadata_y")
    result = dispatch([x*(x*y-1)], x, y)
    assert result.kind == "rational_families"
    assert result.solutions == [] and result.numerators == []
    assert len(result.family_numerators) == len(result.family_denominators) == len(result.family_nonzero_conditions) == 2
    rows = []
    for numerators, denominators, guards in zip(result.family_numerators, result.family_denominators, result.family_nonzero_conditions):
        rows.append(tuple(tuple(_lift_native_result(value, x, y) for value in values) for values in (numerators, denominators, guards)))
    assert set(rows) == {((0, y), (1, 1), ()), ((1, y), (y, 1), (y,))}
    with pytest.raises(AttributeError):
        result.family_nonzero_conditions = []


def test_native_operational_failure_is_not_converted_to_a_partial_answer(monkeypatch):
    x, y = s.symbols("mixed_budget_x mixed_budget_y")

    def exhausted(*args):
        raise ValueError("mixed component budget exhausted")

    monkeypatch.setattr(_native, "polynomial_system_solutions_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        s.nonlinsolve([x*(x*y-1)], x, y)
