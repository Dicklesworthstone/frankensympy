"""Polynomial system solvers for FrankenSymPy (WS19)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple
from ..core import (
    Add,
    Dummy,
    Eq,
    Half,
    Integer,
    Mul,
    NegativeOne,
    One,
    Pow,
    Rational,
    Symbol,
    Zero,
    _lift_native_result,
    _native,
    _native_expr,
    _require_symbol,
    _symbol_binding,
)
from ..sets import EmptySet, FiniteSet, Set


def _polynomial_source(item):
    """Preserve the original exact surface until native polynomial admission.

    In particular a held zero product or Eq must not erase an unsupported
    function, approximate atom, or undefined power before it is checked.
    """
    if type(item) is int:
        item = Integer(item)
    pending = [(item, 0)]
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 4096 or depth > 64:
            raise ValueError("polynomial-system source exceeds its bounded envelope")
        kind = type(node)
        if kind in (Integer, Rational, Zero, One, NegativeOne, Half, Symbol, Dummy):
            continue
        if kind not in (Add, Mul, Pow, Eq):
            raise ValueError("polynomial systems require exact rational polynomial expressions")
        if kind is Eq and node is not item:
            raise ValueError("nested equations are not polynomial expressions")
        children = node.args
        if len(children) > 4096 - visited - len(pending):
            raise ValueError("polynomial-system source exceeds its bounded envelope")
        pending.extend((child, depth + 1) for child in children)
    if type(item) is Eq:
        item = Add(item.lhs, Mul(-1, item.rhs, evaluate=False), evaluate=False)
    return item


def _polynomial_system_inputs(seq, gens):
    """Bound source collection and preserve the caller's typed generator order."""
    if len(gens) == 1 and isinstance(gens[0], (list, tuple)):
        var_list = list(gens[0])
    elif gens:
        var_list = list(gens)
    else:
        var_list = []

    wrapped_eqs = []
    for item in seq:
        if len(wrapped_eqs) == 64:
            raise ValueError("polynomial systems support at most 64 equations")
        wrapped_eqs.append(_polynomial_source(item))

    if not var_list:
        all_syms = set()
        for eq in wrapped_eqs:
            all_syms.update(eq.free_symbols)

        def symbol_order(symbol):
            binding = _symbol_binding(symbol)
            return symbol.name, binding.name, binding.identity_hex

        var_list = sorted(all_syms, key=symbol_order)

    syms = [_require_symbol(v) for v in var_list]
    if len(syms) > 8 or len(set(syms)) != len(syms):
        raise ValueError("polynomial systems require at most eight distinct generators")
    return wrapped_eqs, syms


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
        Complete list of solution tuples corresponding to the ordered
        generators, or ``None`` for an established empty solution set.
        Unsupported, positive-dimensional and undecidable systems raise;
        they are never represented as an empty or partial answer.
    """
    wrapped_eqs, syms = _polynomial_system_inputs(seq, gens)
    if not syms:
        # Validate constant sources through the same bounded native path.
        # Structural inequality with zero cannot decide a held identity, and
        # an undefined held constant must not masquerade as inconsistency.
        syms = [Dummy("_fsym_constant_system")]
    try:
        raw_sols = _native.solve_polynomial_system_native(
            [_native_expr(eq) for eq in wrapped_eqs], [_symbol_binding(v) for v in syms]
        )
    except ValueError as e:
        msg = str(e)
        if "No solution" in msg:
            return None
        if "Infinite solutions" in msg:
            raise NotImplementedError(
                "only zero-dimensional systems supported (finite number of solutions)"
            ) from None
        raise
    if not raw_sols:
        return None
    return [
        tuple(_lift_native_result(val, *wrapped_eqs, *syms) for val in sol)
        for sol in raw_sols
    ]


def nonlinsolve(system: Iterable[Any], *symbols: Any) -> Set:
    """Solve exact rational polynomial systems as sets of solution tuples.

    Affine systems include complete families parameterized by the supplied
    free symbols. Other admitted systems use the complete finite solver;
    unsupported positive-dimensional nonlinear systems remain explicit refusals.
    """
    wrapped_eqs, syms = _polynomial_system_inputs(system, symbols)
    if syms:
        rows = _native.polynomial_affine_matrix_native(
            [_native_expr(eq) for eq in wrapped_eqs], [_symbol_binding(v) for v in syms]
        )
        if rows is not None:
            from ..matrices import Matrix
            from .solvers import linsolve

            # Rational rows go directly to the independent affine checker.
            # No generic expansion, symbolic pivoting, or printer extraction.
            return linsolve(Matrix(rows), *syms)
    sols = solve_poly_system(wrapped_eqs, *syms)
    if not sols:
        return EmptySet()
    return FiniteSet(*sols)


__all__ = [
    "nonlinsolve",
    "solve_poly_system",
]
