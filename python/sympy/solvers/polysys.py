"""Checked polynomial and finite rational system solvers (WS19)."""

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
    """Preserve the original exact surface until native system admission.

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
    if result.kind == "rational_family":
        parts = [(result.numerators, result.denominators, result.nonzero_conditions)]
    elif result.kind == "rational_families":
        count = len(result.family_numerators)
        if not 1 <= count <= 32 or len(result.family_denominators) != count or len(result.family_nonzero_conditions) != count:
            raise ValueError("rational component count does not match the checked union")
        parts = zip(result.family_numerators, result.family_denominators, result.family_nonzero_conditions)
    else:
        parts = None
    if parts is not None:
        return [_lift_rational_component(part, wrapped_eqs, syms) for part in parts]
    if result.kind not in ("real_shape", "real_rur"):
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

    # The native representation proves both solution directions and proves
    # that every complex eliminant root is real before returning either kind.
    # The existing root bridge independently checks factor/index association.
    # No binary64 proximity chooses a root or decides completeness.
    parameter = syms[-1] if result.kind == "real_shape" else Dummy("_fsym_rur_parameter")
    eliminant = Poly(horner(coefficients(result.eliminant), parameter), parameter)
    roots = real_roots(eliminant, multiple=False, radicals=False)
    if [int(multiplicity) for _, multiplicity in roots] != list(result.multiplicities):
        raise ValueError("algebraic root lifting disagrees with the checked complete root set")
    coordinates = [coefficients(values) for values in result.coordinates]
    if len(coordinates) != len(syms):
        raise ValueError("algebraic coordinate count does not match the checked system")
    return [tuple(horner(values, root) for values in coordinates) for root, _ in roots]


def _lift_rational_component(parts, wrapped_eqs, syms):
    """Preserve each checked component's parameter domain independently."""
    raw_numerators, raw_denominators, raw_conditions = parts
    numerators = [_lift_native_result(value, *wrapped_eqs, *syms) for value in raw_numerators]
    denominators = [_lift_native_result(value, *wrapped_eqs, *syms) for value in raw_denominators]
    conditions = [_lift_native_result(value, *wrapped_eqs, *syms) for value in raw_conditions]
    if len(numerators) != len(syms) or len(denominators) != len(syms):
        raise ValueError("rational coordinate count does not match the checked system")
    used_conditions = set()
    values = []
    for numerator, denominator in zip(numerators, denominators):
        if denominator == Integer(1):
            values.append(numerator)
            continue
        if denominator.free_symbols:
            if denominator not in conditions:
                raise ValueError("rational coordinate lost its checked nonzero condition")
            used_conditions.add(denominator)
        else:
            # Exact nonzero constant denominators have no parameter
            # exclusion and can use the ordinary canonical constructor.
            values.append(numerator / denominator)
            continue
        # Normal evaluation may cancel a denominator. Keep it explicitly
        # in the tuple, whose parameter domain excludes its zeros.
        inverse = Pow(denominator, -1, evaluate=False)
        values.append(inverse if numerator == Integer(1) else Mul(numerator, inverse, evaluate=False))
    if used_conditions != set(conditions):
        raise ValueError("rational family omitted a checked parameter exclusion")
    return tuple(values)


def solve_poly_system(
    seq: Iterable[Any], *gens: Any
) -> Optional[List[Tuple[Any, ...]]]:
    """Return complete finite solutions of exact rational polynomial systems.

    Tuples follow the supplied generator order; ``None`` denotes an established
    empty set. Explicit root formulas are preferred. Bounded higher-degree
    systems use exact indexed algebraic roots when a checked original-coordinate
    or separating-linear-form representation has only real eliminant roots.
    Unsupported, positive-dimensional and undecidable systems raise rather than
    returning an empty or partial answer.
    """
    wrapped_eqs, syms = _polynomial_system_inputs(seq, gens)
    return _checked_polynomial_solutions(wrapped_eqs, syms, allow_families=False) or None


def nonlinsolve(system: Iterable[Any], *symbols: Any) -> Set:
    """Solve exact polynomial and supported finite rational systems as tuple sets.

    Affine, polynomial and recognized rational graph systems include complete
    families parameterized by the supplied free symbols. Rational coordinates
    retain their denominator exclusions; their completeness check rules out
    omitted exceptional fibers. Checked factor splits also return complete
    unions of polynomial and rational families, preserving each component's
    parameter exclusions independently. Finite systems use explicit roots
    or checked real algebraic univariate representations. Other unsupported
    positive-dimensional systems remain explicit refusals. Rational source
    equations retain every original denominator exclusion and use a checked
    localization before finite solving. Positive-dimensional rational sources
    remain unsupported when their parameter exclusions cannot be represented.
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
