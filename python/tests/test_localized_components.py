"""Original rational sources project complete guarded families to original symbols."""

import pytest
import sympy as s
from sympy.core import _lift_native_result, _native, _native_expr, _symbol_binding


def ratio(numerator, denominator):
    return s.Mul(numerator, s.Pow(denominator, -1, evaluate=False), evaluate=False)


def solutions(equations, *variables):
    return {tuple(row) for row in s.nonlinsolve(equations, *variables)}


def dispatch(equations, *variables):
    return _native.polynomial_system_solutions_native(
        [_native_expr(value) for value in equations],
        [_symbol_binding(variable) for variable in variables],
        True,
    )


def lift_parts(result, *originals):
    return tuple(
        tuple(_lift_native_result(value, *originals) for value in values)
        for values in (result.numerators, result.denominators, result.nonzero_conditions)
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


def test_reciprocal_source_returns_its_complete_family_in_caller_coordinate_order():
    x, y, z = s.symbols("localized_x localized_y localized_z")
    equation = 1/x-y
    assert solutions([equation], x, y) == {(1/y, y)}
    result = solutions((value for value in [equation]), (y, z, x))
    assert result == {(1/x, z, x)}
    row = next(iter(result))
    assert row[1] is z and row[2] is x
    assert inverse_bases(row[0]) == {x}
    assert row[0].free_symbols == {x}


def test_localized_graph_preserves_complex_poles_and_nonlinear_coordinates():
    x, y, z = s.symbols("localized_curve_x localized_curve_y localized_curve_z")
    result = solutions([1/x-y, z-x*x], x, z, y)
    assert len(result) == 1
    row = next(iter(result))
    assert row[2] is y and inverse_bases(row[0]) == {y}
    for parameter in [-3, -1, 2]:
        values = [value.subs(y, parameter) for value in row]
        assert values == [s.Rational(1, parameter), s.Rational(1, parameter**2), parameter]

    complex_row = next(iter(solutions([1/x-(y*y+1)], x, y)))
    assert complex_row[1] is y
    assert inverse_bases(complex_row[0]) == {y*y+1}
    assert (y*y+1).subs(y, s.I) == 0


def test_localized_factored_union_keeps_both_branches_and_original_poles():
    x, y = s.symbols("localized_union_x localized_union_y")
    equation = (1/x-y)*(1/x+y)
    assert solutions([equation], x, y) == {(1/y, y), (-1/y, y)}
    native = dispatch([equation], x, y)
    assert native.kind == "rational_families"
    assert len(native.family_numerators) == 2
    assert len(native.family_denominators) == len(native.family_nonzero_conditions) == 2
    for denominators, guards in zip(native.family_denominators, native.family_nonzero_conditions):
        assert tuple(_lift_native_result(value, x, y) for value in denominators) == (y, 1)
        assert tuple(_lift_native_result(value, x, y) for value in guards) == (y,)

    # Clearing the numerator alone would introduce the forbidden x = 0 line.
    assert solutions([x*(1/x-y)], x, y) == {(1/y, y)}


def test_native_projection_retains_conditions_used_only_by_discarded_auxiliary():
    x, y = s.symbols("localized_diagonal_x localized_diagonal_y")
    equation = ratio(x-y, x-1)
    native = dispatch([equation], x, y)
    assert native.kind == "rational_family"
    assert lift_parts(native, x, y) == ((y, y), (1, 1), (y-1,))
    with pytest.raises(AttributeError):
        native.nonzero_conditions = []
    # A plain tuple (y, y) would silently include the excluded point (1, 1).
    with pytest.raises(ValueError, match="positive-dimensional rational"):
        s.nonlinsolve([equation], x, y)


def test_zero_products_held_identities_and_zeroth_powers_keep_original_exclusions():
    x, y = s.symbols("localized_held_x localized_held_y")
    reciprocal = s.Pow(x, -1, evaluate=False)
    equations = [
        s.Mul(0, reciprocal, evaluate=False),
        s.Add(s.Mul(x, reciprocal, evaluate=False), -1, evaluate=False),
        s.Add(s.Pow(reciprocal, 0, evaluate=False), -1, evaluate=False),
        s.Eq(reciprocal, reciprocal, evaluate=False),
    ]
    for equation in equations:
        if type(equation) is s.Eq:
            native_source = s.Add(equation.lhs, s.Mul(-1, equation.rhs, evaluate=False), evaluate=False)
        else:
            native_source = equation
        native = dispatch([native_source], x, y)
        assert lift_parts(native, x, y) == ((x, y), (1, 1), (x,))
        with pytest.raises(ValueError, match="positive-dimensional rational"):
            s.nonlinsolve([equation], x, y)


def test_same_printing_assumed_and_dummy_parameters_keep_surface_owners():
    plain = s.Symbol("_fsym_rational_inverse_0")
    assumed = s.Symbol("_fsym_rational_inverse_0", real=True)
    first, second = s.Dummy("localized_same"), s.Dummy("localized_same")
    for dependent, free in [(plain, assumed), (first, second)]:
        row = next(iter(solutions([1/dependent-free], dependent, free)))
        assert row == (1/free, free)
        assert row[1] is free
        assert row[0].free_symbols == {free}
        assert free is not dependent
        assert inverse_bases(row[0]) == {free}


def test_unsupported_component_prevents_any_partial_public_union():
    x, y = s.symbols("localized_partial_x localized_partial_y")
    with pytest.raises((ValueError, NotImplementedError)):
        s.nonlinsolve([(1/x-y)*(x*x+y*y-1)], x, y)


def test_source_guards_are_checked_before_inconsistency_or_solved_branch_shortcuts():
    x, y = s.symbols("localized_invalid_x localized_invalid_y")
    for power in [-1, 0]:
        invalid = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
        for equations in [[1/x-y, invalid], [1, 1/x-y, invalid]]:
            with pytest.raises(ValueError):
                s.nonlinsolve(equations, x, y)


def test_finite_api_remains_polynomial_only_and_finite_localization_keeps_its_shape():
    x, y = s.symbols("localized_finite_x localized_finite_y")
    equations = [ratio(x*x-1, x-1), y-x]
    assert dispatch(equations, x, y).kind == "finite"
    assert solutions(equations, x, y) == {(-1, -1)}
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([1/x-y], x, y)


def test_operational_failure_does_not_restart_or_return_an_unchecked_family(monkeypatch):
    x, y = s.symbols("localized_failure_x localized_failure_y")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("localized component compute budget exhausted")

    monkeypatch.setattr(_native, "polynomial_system_solutions_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        s.nonlinsolve([1/x-y], x, y)
    assert len(calls) == 1
