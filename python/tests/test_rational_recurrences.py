"""Rational recurrence solutions, including shifted and repeated pole chains."""

import pytest
import sympy as s
from sympy.solvers import rsolve_ratio
from sympy.solvers.recurr import rsolve_ratio as ratio_entry


def test_poles_and_polynomial_forcing_are_rational_identities():
    n = s.Symbol("n")
    assert s.rsolve_ratio is ratio_entry is rsolve_ratio
    c = s.Symbol("C3")
    assert rsolve_ratio([-n, n+1], 1, n, symbols=True, shift=3) == (1 + c/n, [c])
    assert rsolve_ratio([n*n+1, 1], 1, n) is None


@pytest.mark.parametrize("base", ["simple", "repeated", "nonreal", "chain"])
def test_complete_first_order_rational_kernels(base):
    n = s.Symbol("n")
    a, b, den = {
        "simple": (n, n+1, n),
        "repeated": (n*n, (n+1)**2, n*n),
        "nonreal": (n*n+1, (n+1)**2+1, n*n+1),
        "chain": (n, n+4, n*(n+1)*(n+2)*(n+3)),
    }[base]
    solution, cs = rsolve_ratio([-a, b], 0, n, symbols=True)
    assert len(cs) == 1
    # Normalization of a free parameter does not change its family.
    kernel = solution.subs(cs[0], 1)
    scale = (den*kernel).subs(n, 1)
    assert scale != 0
    for point in (2, 5, 9, 13, -9):
        assert (den*kernel).subs(n, point) == scale


def test_shifted_unknown_order_zero_and_polynomial_subspace():
    n = s.Symbol("n")
    assert rsolve_ratio([0, n, 0], 3, n) == 3/(n-1)
    assert rsolve_ratio([2*n], n, n, symbols=True) == (s.Rational(1, 2), [])
    assert rsolve_ratio([0, 0, -1, 1], 1, n) == n+s.Symbol("C0")
    assert rsolve_ratio([1, 1], 0, n, symbols=True) == (0, [])


def test_third_order_cubic_endpoints_have_a_nonpolynomial_solution():
    n = s.Symbol("n")
    coeffs = [-2*n**3+n**2+2*n-1, 2*n**3+n**2-6*n,
              -2*n**3-11*n**2-18*n-9, 2*n**3+13*n**2+22*n+8]
    expr, cs = rsolve_ratio(coeffs, 0, n, symbols=True)
    assert len(cs) == 1
    kernel = expr.subs(cs[0], 1)
    for t in (0, 2, 3, 8, -4):
        assert kernel.subs(n, t) == s.Rational(2*t-3, 2*(t*t-1))


def test_an_excessive_pole_chain_is_a_refusal_not_inconsistency():
    n = s.Symbol("n")
    with pytest.raises(NotImplementedError, match="dispersion|degree"):
        rsolve_ratio([-n, n+200], 0, n)


def test_source_replay_rejects_an_incorrect_clearing_multiplier(monkeypatch):
    from sympy.solvers import _rational_recurrence as engine
    n = s.Symbol("n")
    transform = engine.clear_denominators
    def corrupt(*args):
        coeffs, rhs, denominator = transform(*args)
        return coeffs, rhs, {0: next(iter(denominator.values()))}
    monkeypatch.setattr(engine, "clear_denominators", corrupt)
    with pytest.raises(RuntimeError, match="original rational recurrence"):
        rsolve_ratio([-n, n+1], 1, n)
