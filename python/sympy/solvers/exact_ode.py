"""Implicit potentials for bounded exact polynomial differential forms."""

from ..core import Eq, Symbol, _native, _native_expr, _require_symbol, _wrap


def dsolve_exact_first_order(M, N, x, y):
    """Return an implicit relation for the exact form ``M dx + N dy = 0``.

    ``x`` and ``y`` must be distinct Symbol or Dummy objects. ``M`` and ``N``
    must be polynomials over the rationals in those coordinates, with total
    degree at most 16, within the native term and coefficient growth bounds.
    A successful result is an unevaluated ``Eq(F, C)`` whose potential satisfies
    ``diff(F, x) == M`` and ``diff(F, y) == N``. The native checker verifies both
    identities before returning the potential. ``C`` is the first available
    name among C1, C2, ... that occurs in neither the coordinates nor the input.

    The relation describes level sets of F; it does not select a graph y(x),
    impose initial conditions, or assert uniqueness at singular points.
    In particular, a nonzero form with ``N == 0`` may describe vertical curves.
    Nonexact forms, nonpolynomial or inexact coefficients, unsupported sizes,
    and the identically zero form raise ValueError.

    Examples
    ========

    >>> from sympy import Symbol, dsolve_exact_first_order
    >>> x, y = Symbol('x'), Symbol('y')
    >>> relation = dsolve_exact_first_order(2*x + y, x + 2*y, x, y)
    >>> relation.lhs.diff(x) == 2*x + y
    True
    >>> relation.lhs.diff(y) == x + 2*y
    True
    """
    x = _require_symbol(x)
    y = _require_symbol(y)
    m_native = _native_expr(M)
    n_native = _native_expr(N)
    potential = _wrap(
        _native.exact_ode_potential_native(
            m_native, n_native, _native_expr(x), _native_expr(y)
        )
    )
    used_symbols = _wrap(m_native).free_symbols | _wrap(n_native).free_symbols | {x, y}
    used_names = {symbol.name for symbol in used_symbols}
    index = 1
    while f"C{index}" in used_names:
        index += 1
    return Eq(potential, Symbol(f"C{index}"), evaluate=False)


__all__ = ["dsolve_exact_first_order"]
