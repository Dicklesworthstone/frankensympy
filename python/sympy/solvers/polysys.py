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
    elif gens:
        var_list = list(gens)
    else:
        var_list = []

    wrapped_eqs = []
    for item in seq:
        if type(item) is Eq:
            item = item.lhs - item.rhs
        wrapped_eqs.append(_wrap(_native_expr(item)))

    if not var_list:
        all_syms = set()
        for eq in wrapped_eqs:
            all_syms.update(eq.free_symbols)
        var_list = sorted(list(all_syms), key=lambda s: s.name)

    if len(var_list) == 0:
        if not wrapped_eqs or all(eq == 0 for eq in wrapped_eqs):
            raise NotImplementedError(
                "only zero-dimensional systems supported (finite number of solutions)"
            )
        if any(eq != 0 for eq in wrapped_eqs):
            return None
        return []

    if len(var_list) == 1:
        x = _require_symbol(var_list[0])
        candidate_roots = None
        constrained = False
        for eq in wrapped_eqs:
            if eq == 0:
                continue
            if not eq.free_symbols:
                return None
            constrained = True
            try:
                raw = _native.solve_expr(str(eq), _native_symbol_key(x))
            except Exception as e:
                msg = str(e)
                if "No solution" in msg or "Infinite solutions" in msg:
                    return None
                raise
            eq_roots = {_parse_result(r) for r in raw}
            if candidate_roots is None:
                candidate_roots = eq_roots
            else:
                candidate_roots = candidate_roots.intersection(eq_roots)
            if not candidate_roots:
                return None

        if not constrained:
            raise NotImplementedError(
                "only zero-dimensional systems supported (finite number of solutions)"
            )
        if candidate_roots is None:
            return None
        sorted_roots = sorted(list(candidate_roots), key=lambda r: str(r))
        return [(r,) for r in sorted_roots]

    if len(var_list) == 2:
        x = _require_symbol(var_list[0])
        y = _require_symbol(var_list[1])
        eq_list = [str(eq) for eq in wrapped_eqs]
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
        f"solve_poly_system currently supports up to 2-variable systems, got {len(var_list)} generators"
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
