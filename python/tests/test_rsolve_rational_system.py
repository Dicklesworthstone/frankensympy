"""Regular full-dimensional rational fundamental systems through rsolve."""

import pytest
import sympy as s


def recurrence(n, y, weights, start=0, domain=None):
    domain = domain or (lambda t: t*t+1)
    return sum(w*domain(n+j)*y(n+j) for j, w in enumerate(weights, start))


@pytest.mark.parametrize("start", [0, -2, 3])
def test_full_basis_survives_unnormalized_shifts(start):
    n, y = s.Symbol("n"), s.Function("rational_y")
    source = recurrence(n, y, (1, -2, 1), start)
    ans = s.rsolve(source, y(n))
    a, b = s.symbols("C0 C1")
    assert ans.free_symbols == {a, b, n}
    for k in (-3, -1, 0, 2, 7):
        assert s.cancel(ans.subs(n, k)*(k*k+1) - (a+b*k)) == 0


def test_forced_equality_nonconsecutive_initial_values_and_exact_replay():
    n, y = s.Symbol("n"), s.Function("rational_y")
    eq = s.Eq(recurrence(n, y, (1, -2, 1)), 2, evaluate=False)
    ans = s.rsolve(eq, y(n), {y(-1): 0, y(2): s.Rational(9, 5)})
    for k in (-6, -1, 0, 1, 2, 8):
        assert ans.subs(n, k) == s.Rational((k+1)**2, k*k+1)


def test_partial_and_contradictory_conditions():
    n, y = s.Symbol("n"), s.Function("rational_y")
    eq = recurrence(n, y, (1, -2, 1))
    partial = s.rsolve(eq, y(n), {0: 3})
    free = partial.free_symbols - {n}
    assert len(free) == 1
    assert partial.subs(n, 0) == 3
    with pytest.raises(ValueError, match="inconsistent"):
        s.rsolve(eq, y(n), {0: 1, 1: 2, 2: 100})


def test_rank_deficiency_does_not_erase_nonrational_solutions():
    n, y = s.Symbol("n"), s.Function("rational_y")
    # The missing second direction is 2**n/(n**2+1).
    with pytest.raises(NotImplementedError, match="do not span"):
        s.rsolve(recurrence(n, y, (2, -3, 1)), y(n))


def test_singular_lattice_coefficient_is_not_a_global_fundamental_system():
    n, y = s.Symbol("n"), s.Function("rational_y")
    with pytest.raises(NotImplementedError, match="singular"):
        s.rsolve(recurrence(n, y, (1, -2, 1), domain=lambda t: t), y(n))


def test_unshifted_source_replay_rejects_bad_family(monkeypatch):
    from sympy.solvers import _rational_sequence as solver
    n, y = s.Symbol("n"), s.Function("rational_y")
    a, b = s.symbols("C0 C1")
    monkeypatch.setattr(solver, "_solve_recurrence", lambda *args, **kwargs: (a+b*n,[a,b]))
    with pytest.raises(RuntimeError, match="original unshifted recurrence"):
        s.rsolve(recurrence(n, y, (1, -2, 1)), y(n))


def test_basis_dimension_requires_independent_lattice_values(monkeypatch):
    from sympy.solvers import _rational_sequence as solver
    n, y = s.Symbol("n"), s.Function("rational_y")
    a, b = s.symbols("C0 C1")
    duplicate = (a+b)/(n*n+1)
    monkeypatch.setattr(solver, "_solve_recurrence", lambda *args, **kwargs: (duplicate, [a,b]))
    with pytest.raises(RuntimeError, match="dependent"):
        s.rsolve(recurrence(n, y, (1, -2, 1)), y(n))
