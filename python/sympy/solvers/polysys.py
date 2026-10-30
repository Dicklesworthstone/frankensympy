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


def _checked_polynomial_solutions(wrapped_eqs, syms, *, allow_families):
    """Lift one native outcome without restarting a failed algorithm's budget."""
    if not syms:
        # Validate constant sources through the same bounded native path.
        # Structural inequality with zero cannot decide a held identity, and
        # an undefined held constant must not masquerade as inconsistency.
        syms = [Dummy("_fsym_constant_system")]
        allow_families = False
    try:
        result = _native.polynomial_system_solutions_native(
            [_native_expr(eq) for eq in wrapped_eqs],
            [_symbol_binding(v) for v in syms],
            allow_families,
        )
    except ValueError as e:
        msg = str(e)
        if "No solution" in msg:
            return []
        if "Infinite solutions" in msg:
            raise NotImplementedError(
                "this positive-dimensional system has no supported complete parameterization"
            ) from None
        raise
    if result.kind in ("finite", "family"):
        return [
            tuple(_lift_native_result(val, *wrapped_eqs, *syms) for val in sol)
            for sol in result.solutions
        ]
    if result.kind != "real_shape":
        raise ValueError("unrecognized native polynomial-system outcome")

    from ..polys.polytools import Poly
    from ..polys.rootoftools import real_roots

    def coefficients(values):
        return [_lift_native_result(value) for value in values]

    def horner(values, parameter):
        value = Integer(0)
        for coefficient in reversed(values):
            value = value * parameter + coefficient
        return value

    # The native shape checker proves both solution directions and proves
    # that every complex eliminant root is real before returning this kind.
    # The existing root bridge independently checks factor/index association.
    # No binary64 proximity chooses a root or decides completeness.
    eliminant = Poly(horner(coefficients(result.eliminant), syms[-1]), syms[-1])
    roots = real_roots(eliminant, multiple=False, radicals=False)
    if [int(multiplicity) for _, multiplicity in roots] != list(result.multiplicities):
        raise ValueError("algebraic root lifting disagrees with the checked complete root set")
    coordinates = [coefficients(values) for values in result.coordinates]
    if len(coordinates) != len(syms):
        raise ValueError("algebraic coordinate count does not match the checked system")
    return [tuple(horner(values, root) for values in coordinates) for root, _ in roots]


def solve_poly_system(
    seq: Iterable[Any], *gens: Any
) -> Optional[List[Tuple[Any, ...]]]:
    """Return complete finite solutions of exact rational polynomial systems.

    Tuples follow the supplied generator order; ``None`` denotes an established
    empty set. Explicit root formulas are preferred. Higher-degree shape systems
    whose eliminant roots are all real use exact indexed algebraic roots.
    Unsupported, positive-dimensional and undecidable systems raise rather than
    returning an empty or partial answer.
    """
    wrapped_eqs, syms = _polynomial_system_inputs(seq, gens)
    return _checked_polynomial_solutions(wrapped_eqs, syms, allow_families=False) or None


def nonlinsolve(system: Iterable[Any], *symbols: Any) -> Set:
    """Solve exact rational polynomial systems as sets of solution tuples.

    Affine and recognized polynomial graph systems include complete families
    parameterized by the supplied free symbols. Checked factor splits also
    return complete unions of such families. Finite systems use explicit roots
    or checked real algebraic shape representations. Other unsupported
    positive-dimensional systems remain explicit refusals.
    """
    wrapped_eqs, syms = _polynomial_system_inputs(system, symbols)
    sols = _checked_polynomial_solutions(wrapped_eqs, syms, allow_families=True)
    if not sols:
        return EmptySet()
    from ..core import Tuple as SymTuple

    return FiniteSet(*(SymTuple(*solution) for solution in sols))


__all__ = [
    "nonlinsolve",
    "solve_poly_system",
]
