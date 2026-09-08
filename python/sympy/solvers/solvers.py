"""General equation solvers and linear system solvers for FrankenSymPy (WS19)."""

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
from ..core import (
    Basic,
    Eq,
    Expr,
    Integer,
    Rational,
    Symbol,
    _native_expr,
    _require_symbol,
    _wrap,
)
from ..matrices import Matrix, MatrixBase
from ..sets import EmptySet, FiniteSet, Set
from ..core import expand, simplify


def _extract_linear_coeffs(expr, symbols):
    """Given an expression expr and a list of symbols, return (coeffs, const)
    such that expr == sum(c * s for c, s in zip(coeffs, symbols)) + const.
    If expr is not linear in symbols, return None.
    """
    expr = expand(expr)
    zero_subs = {s: 0 for s in symbols}
    const = expr.subs(zero_subs)
    coeffs = []
    for s in symbols:
        subs_one = dict(zero_subs)
        subs_one[s] = 1
        c = expr.subs(subs_one) - const
        coeffs.append(c)

    # Verify linearity: expr - (sum(coeffs * s) + const) should be 0
    recon = const
    for c, s in zip(coeffs, symbols):
        recon = recon + c * s
    diff = expand(expr - recon)
    if diff != 0:
        return None
    return coeffs, const


def _solve_augmented_matrix_to_set(M: Matrix, sym_list: List[Symbol]) -> Set:
    """Solve augmented matrix M using RREF and return FiniteSet or EmptySet."""
    n = M.cols - 1
    if n != len(sym_list):
        raise ValueError(f"Matrix columns - 1 ({n}) does not match symbols ({len(sym_list)})")

    rref_mat, pivots = M.rref()
    # Check consistency: if the last column (index n) is a pivot, system is inconsistent (0 == 1)
    if n in pivots:
        return EmptySet()

    # Map pivot column -> row index
    pivot_row_map = {col: r for r, col in enumerate(pivots) if col < n}
    free_cols = [c for c in range(n) if c not in pivot_row_map]

    sol = []
    for c in range(n):
        if c in pivot_row_map:
            r = pivot_row_map[c]
            val = rref_mat[r, n]
            for fc in free_cols:
                coeff = rref_mat[r, fc]
                if coeff != 0:
                    val = val - coeff * sym_list[fc]
            sol.append(simplify(val))
        else:
            # Free variable
            sol.append(sym_list[c])

    return FiniteSet(tuple(sol))


def linsolve(system: Any, *symbols: Any) -> Set:
    """Solve linear system of equations.

    Parameters
    ----------
    system : list/tuple of equations, augmented Matrix, or (A, b) pair
        The linear system to solve.
    *symbols : Symbol or Sequence of Symbol
        The symbols to solve for.

    Returns
    -------
    FiniteSet
        A FiniteSet containing a tuple of solutions, or EmptySet if inconsistent.
    """
    # Parse symbols
    if len(symbols) == 1 and isinstance(symbols[0], (list, tuple)):
        sym_list = list(symbols[0])
    elif symbols:
        sym_list = list(symbols)
    else:
        sym_list = []

    # Determine input form
    if (
        isinstance(system, (tuple, list))
        and len(system) == 2
        and isinstance(system[0], (MatrixBase, list))
        and isinstance(system[1], (MatrixBase, list))
    ):
        A = Matrix(system[0])
        b_input = system[1]
        if not isinstance(b_input, MatrixBase):
            b = Matrix(b_input)
        else:
            b = b_input
        if b.rows == 1 and b.cols == A.rows:
            b = b.T
        if b.cols != 1 or b.rows != A.rows:
            raise ValueError("Incompatible dimensions for (A, b) in linsolve")
        M = A.row_join(b)
        n = A.cols
        if not sym_list:
            sym_list = [Symbol(f"x{i}") for i in range(n)]
        elif len(sym_list) != n:
            raise ValueError(f"Number of symbols ({len(sym_list)}) does not match system columns ({n})")
    elif isinstance(system, MatrixBase):
        M = system
        n = M.cols - 1
        if not sym_list:
            sym_list = [Symbol(f"x{i}") for i in range(n)]
        elif len(sym_list) != n:
            raise ValueError(f"Number of symbols ({len(sym_list)}) does not match matrix cols - 1 ({n})")
    elif isinstance(system, (list, tuple)):
        if system and isinstance(system[0], (list, tuple)):
            M = Matrix(system)
            n = M.cols - 1
            if not sym_list:
                sym_list = [Symbol(f"x{i}") for i in range(n)]
            elif len(sym_list) != n:
                raise ValueError(f"Number of symbols ({len(sym_list)}) does not match matrix cols - 1 ({n})")
        else:
            # List of equations or expressions
            eq_list = []
            for item in system:
                if type(item) is Eq:
                    item = item.lhs - item.rhs
                eq_list.append(_wrap(_native_expr(item)))

            if not sym_list:
                all_syms = set()
                for eq in eq_list:
                    all_syms.update(eq.free_symbols)
                sym_list = sorted(list(all_syms), key=lambda s: s.name)

            n = len(sym_list)
            rows = []
            for eq in eq_list:
                res = _extract_linear_coeffs(eq, sym_list)
                if res is None:
                    raise ValueError(f"Equation {eq} is not linear in symbols {sym_list}")
                coeffs, const = res
                # eq = sum(coeffs * syms) + const = 0 => sum(coeffs * syms) = -const
                row = coeffs + [-const]
                rows.append(row)

            if not rows:
                return FiniteSet(tuple(sym_list))
            M = Matrix(rows)
    else:
        raise TypeError(f"Invalid system type: {type(system)}")

    return _solve_augmented_matrix_to_set(M, sym_list)


def solve_linear_system(system: MatrixBase, *symbols: Any) -> Optional[dict]:
    """Solve an augmented matrix linear system.

    Parameters
    ----------
    system : Matrix
        The augmented matrix [A | b] representing A * x = b.
    *symbols : Symbol or Sequence of Symbol
        The symbols to solve for.

    Returns
    -------
    dict or None
        A dictionary mapping each symbol to its solution, or None if inconsistent.
    """
    if len(symbols) == 1 and isinstance(symbols[0], (list, tuple)):
        sym_list = list(symbols[0])
    elif symbols:
        sym_list = list(symbols)
    else:
        raise ValueError("solve_linear_system requires symbols")

    if not isinstance(system, MatrixBase):
        system = Matrix(system)

    sols_set = _solve_augmented_matrix_to_set(system, sym_list)
    if not sols_set or isinstance(sols_set, EmptySet):
        return None
    sol_tuple = next(iter(sols_set))
    # RREF keeps free parameters in the solution tuple; this API reports
    # assignments only, so an unconstrained symbol must not map to itself.
    return {sym: val for sym, val in zip(sym_list, sol_tuple) if sym != val}


solve_linear_system_LU = solve_linear_system


def solve_linear(lhs: Any, rhs: Any = 0, symbols: Any = (), exclude: Any = ()) -> Tuple[Any, Any]:
    """Return a tuple derived from f = lhs - rhs: (symbol, solution), (0, 1), (0, 0), or (n, d)."""
    if type(lhs) is Eq:
        if rhs != 0:
            raise ValueError(f"If lhs is an Equality, rhs must be 0 but was {rhs}")
        f = lhs.lhs - lhs.rhs
    else:
        f = lhs - rhs

    f_expr = _wrap(_native_expr(f))
    free = f_expr.free_symbols

    target_symbols = list(symbols) if symbols else sorted(list(free), key=lambda s: s.name)
    exclude_set = set(exclude) if exclude else set()
    target_symbols = [s for s in target_symbols if s not in exclude_set]

    if not target_symbols:
        return (0, 1)

    for sym in target_symbols:
        res = _extract_linear_coeffs(f_expr, [sym])
        if res is not None:
            coeffs, const = res
            a = coeffs[0]
            if a != 0:
                sol = simplify(-const / a)
                return (sym, sol)

    return (f_expr, 1)


__all__ = [
    "linsolve",
    "solve_linear",
    "solve_linear_system",
    "solve_linear_system_LU",
]
