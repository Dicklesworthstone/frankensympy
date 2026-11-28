"""Polynomial solution spaces of variable-coefficient linear recurrences."""

from fractions import Fraction

import pytest
import sympy as s
from sympy.solvers import rsolve_poly
from sympy.solvers.recurr import rsolve_poly as recurrence_poly


def test_public_entries_and_discrete_antiderivatives():
    assert s.rsolve_poly is rsolve_poly is recurrence_poly
    n = s.Symbol('n')
    assert rsolve_poly([-1, 1], n, n) == s.Symbol('C0') + n**2/2 - n/2
    assert rsolve_poly([-1, 1], n**3, n, shift=3, symbols=True) == (
        s.Symbol('C3') + n**4/4 - n**3/2 + n**2/4, [s.Symbol('C3')])


def test_second_difference_keeps_both_free_directions():
    n = s.Dummy('n')
    result, constants = rsolve_poly([1, -2, 1], 0, n, shift=2, symbols=True)
    assert constants == list(s.symbols('C2 C3'))
    assert result == constants[0] + constants[1]*n


@pytest.mark.parametrize("degree", [2, 7, 16, 41, 64])
def test_resonance_finds_high_degree_polynomial_solutions(degree):
    # (n+d)*u(n) - n*u(n+1) = 0. The unique monic polynomial
    # solution is the rising factorial of n, with degree d.
    n = s.Symbol('n')
    expr, constants = rsolve_poly([n+degree, -n], 0, n, symbols=True)
    assert len(constants) == 1
    kernel = expr.subs(constants[0], 1)
    assert s.Poly(kernel, n).degree() == degree
    for point in (-3, 0, 1, 5):
        expected = 1
        for k in range(degree):
            expected *= point+k
        assert kernel.subs(n, point) == expected


def test_nonconstant_coefficients_rational_particular_and_homogeneous_family():
    n = s.Symbol('n')
    c = s.Symbol('C0')
    assert rsolve_poly([n+1, -n], s.Rational(3, 7), n) == c*n+s.Rational(3, 7)
    assert rsolve_poly([n+1, -n], n, n, symbols=True) is None
    assert rsolve_poly([2*n+1, n*n+1], 3*(n*n+2*n+2), n, symbols=True) == (3, [])


def test_zero_missing_shifts_zero_operator_and_order_zero():
    n = s.Symbol('n')
    assert rsolve_poly([0, 1, 0, 0], n**2, n) == n**2-2*n+1
    assert rsolve_poly([2*n], n, n) == s.Rational(1, 2)
    assert rsolve_poly([2*n], 1, n) is None
    assert rsolve_poly([0, 0], n, n) is None
    with pytest.raises(NotImplementedError, match="unbounded polynomial freedom"):
        rsolve_poly([0, 0], 0, n)


def test_homogeneous_zero_is_a_solution_and_names_do_not_capture_the_index():
    n = s.Symbol('C0')
    assert rsolve_poly([2, 1], 0, n, symbols=True) == (0, [])
    assert rsolve_poly([-1, 1], 1, n, symbols=True) == (s.Symbol('C1')+n, [s.Symbol('C1')])


def test_hidden_poles_and_distinct_coordinate_identities_are_refused():
    n = s.Symbol('n')
    hidden = s.Mul(0, s.Pow(n, -1, evaluate=False), evaluate=False)
    for coeffs, rhs in (([hidden, 1], 1), ([1], hidden), ([s.Symbol('n', positive=True)], 0),
                        ([s.Float(0.5)], 1)):
        with pytest.raises((TypeError, ValueError, NotImplementedError)):
            rsolve_poly(coeffs, rhs, n)


def test_resource_refusals_do_not_become_no_polynomial_solution():
    n = s.Symbol('n')
    for coeffs, forcing in (([n+65, -n], 0), ([-1, 1], n**64), ([1], n**65)):
        with pytest.raises(NotImplementedError, match="degree"):
            rsolve_poly(coeffs, forcing, n)
    with pytest.raises(ValueError, match="coefficients"):
        rsolve_poly([1]*34, n, n)


def test_corrupt_particular_and_kernel_are_independently_rejected(monkeypatch):
    from sympy.solvers import _polynomial_recurrence as solver
    n = s.Symbol('n')
    for invalid in (({0: Fraction(1)}, []), ({1: Fraction(1)}, [{1: Fraction(3)}])):
        monkeypatch.setattr(solver, '_solve_columns', lambda *args: invalid)
        with pytest.raises(RuntimeError, match="failed the original recurrence"):
            rsolve_poly([-1, 1], 1, n)


def test_mutating_source_list_at_publication_discards_the_result(monkeypatch):
    from sympy.solvers import _polynomial_recurrence as solver
    n = s.Symbol('n')
    coefficients = [-1, 1]
    verify = solver._verify
    def mutate(*args):
        verify(*args)
        coefficients[0] = 3
    monkeypatch.setattr(solver, '_verify', mutate)
    with pytest.raises(RuntimeError, match="coefficient list changed"):
        rsolve_poly(coefficients, 1, n)
