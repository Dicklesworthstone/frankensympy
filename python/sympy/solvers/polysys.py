"""Polynomial system solvers for FrankenSymPy (WS19)."""

from typing import Any, Iterable, List, Optional, Sequence, Tuple
from ..core import (
    Eq,
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)
from ..sets import EmptySet, FiniteSet, Set


def solve_poly_system(
    seq: Iterable[Any], *gens: Any
) -> Optional[List[Tuple[Any, ...]]]:
    """Solve a system of polynomial equations using Lexicographical Groebner bases.

    Parameters
    ----------
    seq : iterable of Expr or Eq
        The polynomial equations or expressions to solve.
    *gens : Symbol or Sequence of Symbol
        The generator variables to solve for.

    Returns
    -------
    list of tuple or None
        List of solution tuples corresponding to the ordered generators,
        or ``None`` if the system has no solutions or cannot be solved.
    """
    if len(gens) == 1 and isinstance(gens[0], (list, tuple)):
        var_list = list(gens[0])
    else:
        var_list = list(gens)

    eq_list = []
    for item in seq:
        if type(item) is Eq:
            item = item.lhs - item.rhs
        eq_list.append(str(_wrap(_native_expr(item))))

    if len(var_list) == 2:
        x = _require_symbol(var_list[0])
        y = _require_symbol(var_list[1])
        try:
            raw_sols = _native.solve_poly_system_expr(
                eq_list, _native_symbol_key(x), _native_symbol_key(y)
            )
        except ValueError as e:
            if "No solution" in str(e):
                return None
            raise
        if not raw_sols:
            return None
        return [
            tuple(_parse_result(val) for val in sol) for sol in raw_sols
        ]

    raise NotImplementedError(
        f"solve_poly_system currently supports 2-variable systems, got {len(var_list)} generators"
    )


def nonlinsolve(system: Iterable[Any], *symbols: Any) -> Set:
    """Solve a system of non-linear equations, returning a FiniteSet of solution tuples."""
    sols = solve_poly_system(system, *symbols)
    if not sols:
        return EmptySet()
    return FiniteSet(*sols)


__all__ = [
    "nonlinsolve",
    "solve_poly_system",
]
