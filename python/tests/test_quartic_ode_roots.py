"""Exact scalar ODE bases for nested complex characteristic radicals.

These regressions require FrankenSymPy's actual shell and root solver. The
counterexample sqrt(1 + I) and the Euler coefficients were independently
checked with isolated SymPy 1.14.0; upstream is not a production fallback.
"""

import pytest
import sympy as s

from sympy.solvers.ode import (
    _assemble,
    _characteristic_root_groups,
    _char_solutions,
    _euler_solutions,
    _homogeneous_basis,
)


def nested_roots():
    return (s.sqrt(1 + s.I), -s.sqrt(1 + s.I), s.sqrt(1 - s.I), -s.sqrt(1 - s.I))


def assert_zero(expression):
    assert s.simplify(s.expand(expression)) == 0


def test_nested_radicals_keep_exact_roots_and_multiplicities():
    # I -> 0 gives 1 for sqrt(1 + I), which is not its real part.
    roots = {root: 2 for root in nested_roots()}
    groups = _characteristic_root_groups(roots)
    assert groups == [("root", root, 2) for root in roots]
    assert all(s.expand(root**4 - 2*root**2 + 2) == 0 for root in roots)


@pytest.mark.parametrize("roots", [
    {1 + 2*s.I: 1},
    {1 + 2*s.I: 2, 1 - 2*s.I: 1},
])
def test_a_missing_or_different_multiplicity_conjugate_is_not_invented(roots):
    assert _characteristic_root_groups(roots) == [
        ("root", root, multiplicity) for root, multiplicity in roots.items()
    ]


def test_an_undecided_imaginary_part_does_not_prove_pair_independence():
    frequency = s.Symbol("ode_unknown_frequency", real=True)
    roots = {s.I*frequency: 1, -s.I*frequency: 1}
    assert _characteristic_root_groups(roots) == [("root", root, 1) for root in roots]


@pytest.mark.parametrize("coefficients,make_expected", [
    ([1, 0, 1], lambda x: {s.sin(x), s.cos(x)}),
    ([5, -2, 1], lambda x: {s.exp(x)*s.sin(2*x), s.exp(x)*s.cos(2*x)}),
    ([-2, 0, 1], lambda x: {s.exp(-s.sqrt(2)*x), s.exp(s.sqrt(2)*x)}),
    ([1, -2, 1], lambda x: {s.exp(x), x*s.exp(x)}),
    ([1, 0, 2, 0, 1], lambda x: {s.sin(x), s.cos(x), x*s.sin(x), x*s.cos(x)}),
])
def test_established_real_and_trigonometric_bases_are_preserved(coefficients, make_expected):
    x = s.Symbol("ode_simple_root_x")
    basis, _ = _char_solutions(coefficients, x)
    assert set(basis) == make_expected(x)


def test_public_constant_coefficient_quartic_has_four_valid_independent_coordinates():
    x = s.Symbol("ode_quartic_x")
    f = s.Function("quartic_solution")(x)
    equation = s.diff(f, x, 4) - 2*s.diff(f, x, 2) + 2*f
    solution = s.dsolve(equation, f)
    constants = {s.Symbol(f"C{index}") for index in range(1, 5)}
    assert solution.lhs == f
    assert solution.rhs.free_symbols == constants | {x}
    basis = [s.diff(solution.rhs, constant) for constant in constants]
    assert len(set(basis)) == 4
    for value in basis:
        assert_zero(s.diff(value, x, 4) - 2*s.diff(value, x, 2) + 2*value)


def test_public_euler_quartic_keeps_the_same_exact_roots_on_the_positive_axis():
    x = s.Symbol("ode_quartic_positive_x", positive=True)
    f = s.Function("quartic_euler_solution")(x)
    # r^4 - 2*r^2 + 2 in the falling-factorial basis.
    coefficients = [2, -1, 5, 6, 1]
    equation = sum(coefficient*x**order*s.diff(f, x, order)
                   for order, coefficient in enumerate(coefficients))
    solution = s.dsolve(equation, f)
    constants = {s.Symbol(f"C{index}") for index in range(1, 5)}
    assert solution.lhs == f
    assert solution.rhs.free_symbols == constants | {x}
    basis = [s.diff(solution.rhs, constant) for constant in constants]
    assert len(set(basis)) == 4
    for value in basis:
        assert_zero(sum(coefficient*x**order*s.diff(value, x, order)
                        for order, coefficient in enumerate(coefficients)))


def test_repeated_nested_quartics_keep_all_polynomial_and_logarithmic_factors():
    x = s.Symbol("ode_repeated_quartic_x", positive=True)
    # (r^4 - 2*r^2 + 2)^2, in ordinary and falling-factorial coordinates.
    coefficients = [4, 0, -8, 0, 8, 0, -4, 0, 1]
    euler_coefficients = [4, -3, 51, 654, 1449, 990, 262, 28, 1]
    exponential, _ = _char_solutions(coefficients, x)
    powers = _euler_solutions(euler_coefficients, x)
    assert len(exponential) == len(powers) == 8
    assert len(set(exponential)) == len(set(powers)) == 8
    exact_groups = _characteristic_root_groups({root: 2 for root in nested_roots()})
    _, exact_basis = _assemble(exact_groups, x)
    assert set(exact_basis) == {x**order*s.exp(root*x)
                                for root in nested_roots() for order in range(2)}
    for value in powers:
        assert_zero(sum(coefficient*x**order*s.diff(value, x, order)
                        for order, coefficient in enumerate(euler_coefficients)))


def test_general_quartic_root_formula_reaches_the_exact_exponential_basis():
    x, r = s.symbols("ode_general_quartic_x ode_characteristic_root")
    roots = s.roots(r**4 + r + 1, r)
    basis, _ = _char_solutions([1, 1, 0, 0, 1], x)
    assert len(roots) == len(basis) == 4
    assert set(basis) == {s.exp(root*x) for root in roots}


def test_legacy_homogeneous_assembly_retains_unsplit_complex_roots():
    x = s.Symbol("ode_legacy_quartic_x")
    groups = _homogeneous_basis([2, 0, -2, 0, 1], x)
    solution, basis = _assemble(groups, x)
    assert len(basis) == 4
    assert solution.free_symbols == {x, *(s.Symbol(f"C{index}") for index in range(1, 5))}
    for value in basis:
        assert_zero(s.diff(value, x, 4) - 2*s.diff(value, x, 2) + 2*value)
