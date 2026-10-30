"""Finite rational systems retain original poles through exact localization."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def ratio(numerator, denominator):
    return s.Mul(numerator, s.Pow(denominator, -1, evaluate=False), evaluate=False)


def points(equations, *variables):
    return {tuple(row) for row in s.nonlinsolve(equations, *variables)}


def dispatch(equations, *variables):
    return _native.polynomial_system_solutions_native(
        [_native_expr(value) for value in equations],
        [_symbol_binding(variable) for variable in variables],
        True,
    )


def quintic(value):
    return value**5 + value**4 - 4*value**3 - 3*value**2 + 3*value + 1


def test_cancelled_original_denominator_removes_only_the_pole():
    x, y = s.symbols("rational_source_x rational_source_y")
    equations = [ratio(x*x-1, x-1), y-x]
    assert points(equations, x, y) == {(-1, -1)}
    assert points((item for item in equations), (y, x)) == {(-1, -1)}


def test_all_original_denominators_constrain_the_same_complete_solution_set():
    x, y = s.symbols("rational_guards_x rational_guards_y")
    equations = [ratio((x-1)*(x+1)*(x-2), x-1), ratio(y-x, x+1)]
    assert points(equations, x, y) == {(2, 2)}


def test_nested_inverse_keeps_its_original_nonzero_domain():
    x = s.Symbol("rational_nested_x")
    nested = s.Pow(s.Pow(x, -1, evaluate=False), -1, evaluate=False)
    assert s.nonlinsolve([nested], x) == s.EmptySet()


def test_zero_products_and_zeroth_powers_do_not_remove_original_poles():
    x, y = s.symbols("rational_zero_x rational_zero_y")
    hidden = s.Mul(0, s.Pow(y, -1, evaluate=False), evaluate=False)
    assert points([s.Add(x, hidden, evaluate=False), y*(y-1)], x, y) == {(0, 1)}
    zero = s.Add(s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False), -1, evaluate=False)
    assert points([zero, x*(x-1)], x) == {(1,)}


def test_held_equalities_preserve_the_domain_of_both_original_sides():
    x = s.Symbol("rational_relation_x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    relation = s.Eq(reciprocal, reciprocal, evaluate=False)
    assert points([relation, x*(x-1)], x) == {(1,)}


def test_rational_sums_and_requested_coordinate_order_remain_exact():
    x, y = s.symbols("rational_order_x rational_order_y")
    equations = [1/x + 1/y - 3, y-1]
    assert points(equations, x, y) == {(s.Rational(1, 2), 1)}
    assert points(equations, y, x) == {(1, s.Rational(1, 2))}


def test_radical_and_nonreal_finite_points_are_not_discarded():
    x, y = s.symbols("rational_complex_x rational_complex_y")
    assert points([ratio(x*x-2, x-1), y-x], x, y) == {
        (-s.sqrt(2), -s.sqrt(2)), (s.sqrt(2), s.sqrt(2)),
    }
    assert points([ratio(x*x+1, x), y-x], x, y) == {(-s.I, -s.I), (s.I, s.I)}


def test_same_printing_and_dummy_generators_keep_distinct_typed_coordinates():
    plain = s.Symbol("rational_identity")
    assumed = s.Symbol("rational_identity", real=True)
    assert points([ratio(plain-1, assumed), assumed-2], assumed, plain) == {(2, 1)}
    first, second = s.Dummy("rational_aux"), s.Dummy("rational_aux")
    assert points([ratio(first-3, second), second-4], first, second) == {(3, 4)}
    with pytest.raises(ValueError):
        s.nonlinsolve([ratio(plain-1, assumed), assumed-2], plain)


def test_empty_rational_locus_is_an_exact_empty_set():
    x, y = s.symbols("rational_empty_x rational_empty_y")
    assert s.nonlinsolve([ratio(x, x), y], x, y) == s.EmptySet()
    assert s.nonlinsolve([ratio(x-1, y), x-2, y-1], x, y) == s.EmptySet()


@pytest.mark.parametrize("power", [-1, 0])
def test_undefined_originals_are_rejected_before_inconsistency_or_zero_shortcuts(power):
    x, y = s.symbols("rational_bad_x rational_bad_y")
    bad = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
    for equations in [[ratio(x-1, y), bad], [1, ratio(x-1, y), bad]]:
        with pytest.raises(ValueError):
            s.nonlinsolve(equations, x, y)


def test_punctured_positive_dimensional_domains_are_explicitly_unsupported():
    x, y = s.symbols("rational_punctured_x rational_punctured_y")
    for equation in [ratio(x, y), s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)]:
        with pytest.raises(ValueError, match="positive-dimensional rational"):
            s.nonlinsolve([equation], x, y)


def test_localization_auxiliary_never_leaks_into_original_coordinate_metadata():
    x, y = s.symbols("rational_projection_x rational_projection_y")
    equations = [ratio(x*x-1, x-1), y-x]
    result = dispatch(equations, x, y)
    assert result.kind == "finite"
    assert len(result.solutions) == 1
    assert len(result.solutions[0]) == 2
    assert all(not value.free_symbols for row in s.nonlinsolve(equations, x, y) for value in row)


def test_all_real_quintic_uses_a_bound_rur_parameter_and_returns_all_five_points():
    x, y = s.symbols("rational_quintic_x rational_quintic_y")
    equations = [ratio(quintic(x), x), y-x]
    result = dispatch(equations, x, y)
    assert result.kind == "real_rur"
    assert len(result.coordinates) == 2
    assert result.multiplicities == [1]*5
    solutions = points(equations, x, y)
    assert len(solutions) == 5
    for first, second in solutions:
        assert first == second
        assert not first.free_symbols and not second.free_symbols
        assert s.Eq(quintic(first), 0) is s.S.true


def test_real_rur_subset_cannot_claim_the_complete_complex_solution_set():
    x, y = s.symbols("rational_nonreal_x rational_nonreal_y")
    with pytest.raises(ValueError):
        s.nonlinsolve([ratio(x**5-x+1, x), y-x], x, y)


def test_constant_reciprocals_preserve_affine_families_and_polynomial_limits():
    x, y = s.symbols("rational_constant_x rational_constant_y")
    constant = s.Pow(s.Integer(2), -1, evaluate=False)
    equation = s.Add(x, s.Mul(constant, y, evaluate=False), evaluate=False)
    assert points([equation], x, y) == {(-y/2, y)}
    variables = s.symbols("rational_polynomial_limit0:8")
    assert points(list(variables), *variables) == {tuple(s.Integer(0) for _ in variables)}


def test_finite_polynomial_api_remains_polynomial_only():
    x = s.Symbol("rational_contract_x")
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([ratio(x*x-1, x-1)], x)


def test_rational_envelope_reserves_room_for_its_auxiliary_equation_and_variable():
    variables = s.symbols("rational_bounded0:8")
    with pytest.raises(ValueError):
        s.nonlinsolve([ratio(variables[0]-1, variables[1])], *variables)
    x = variables[0]
    with pytest.raises(ValueError):
        s.nonlinsolve([ratio(x-1, x)]*64, x)


def test_operational_failure_does_not_restart_or_fall_back_to_polynomial_solving(monkeypatch):
    x = s.Symbol("rational_failure_x")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("rational localization compute budget exhausted")

    monkeypatch.setattr(_native, "polynomial_system_solutions_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        s.nonlinsolve([ratio(x-1, x)], x)
    assert len(calls) == 1
