"""Formal spectra of upper and lower triangular matrices over QQ[parameters]."""

from ..core import (
    Dummy, Pow, Rational, Symbol, _active_callback_session, _charge_nested_if_active,
    _surface_binding_key, _symbol_binding, _wrap,
)
from ..polys import _coefficient_queries
from ..solvers import _sparse_linear as symbols
from . import _definiteness as source
from . import _pseudoinverse
from .sparse import _ZERO


def _polynomial_key(raw, session):
    """An exact coefficient map, independent of held or expanded presentation."""
    if raw.is_integer or raw.is_rational:
        numerator = raw.exact_numerator()
        return (((), numerator, raw.exact_denominator()),) if numerator else ()
    bindings = raw.free_symbol_bindings
    if not bindings:
        bindings = [_symbol_binding(Symbol("_fsym_triangular_parameter"))]
    keys = [_surface_binding_key(binding) for binding in bindings]
    terms = _coefficient_queries._project(raw, bindings, "QQ", "lex", session)
    return tuple(
        (tuple((key, exponent) for key, exponent in zip(keys, powers) if exponent),
         coefficient.exact_numerator(), coefficient.exact_denominator())
        for powers, coefficient in terms
    )


def symbolic_triangular_spectrum(matrix, dense):
    """Check the original orientation and count the original diagonal polynomials."""
    session = _active_callback_session()
    ingress = _pseudoinverse._ingress_snapshot(matrix, "symbolic triangular spectrum")
    if ingress is None:
        raise NotImplementedError("symbolic triangular spectrum requires an original built-in matrix")
    _, base, state, retained = ingress
    native, (n, _), _, _ = state
    if native is None:
        raise NotImplementedError("symbolic triangular spectrum requires closed polynomial cells")
    source._interfaces()
    if retained is None:
        raw = source._NATIVE_CELLS[base][1](native)
        total = 0
        for value in raw if dense else raw.values():
            total += source._POLYNOMIAL_NATIVE[1](value)
            if total > 131_072:
                raise ValueError("triangular matrix exceeds the polynomial source bound")
        retained = (tuple(_wrap(value) for value in raw) if dense else
                    {key: _wrap(value) for key, value in raw.items()})
    cells = ({divmod(i, n): v for i, v in enumerate(retained) if v is not _ZERO}
             if dense else retained)
    above = any(i < j and v is not _ZERO for (i, j), v in cells.items())
    below = any(i > j and v is not _ZERO for (i, j), v in cells.items())
    if above and below:
        return None

    # Bind every original coordinate and every distinct retained scalar owner.
    # A held zero on the unused side cannot hide a pole or a foreign object.
    _charge_nested_if_active(1 + len(retained))
    unique = {id(v): v for v in cells.values()}
    unique[id(_ZERO)] = _ZERO
    scalars = source._closed(unique.values(), polynomial=True)
    if scalars is None:
        raise NotImplementedError("triangular eigenvalues require closed commuting polynomial entries")
    source._bind_gaussian(native, base, retained, scalars, polynomial=True)
    symbols._interfaces()
    records = []
    seen = set()
    for value, _, stored, _, _ in scalars:
        if type(value) in (Symbol, Dummy) and id(value) not in seen:
            seen.add(id(value))
            records.extend(symbols._coordinates((value,), 1))
        elif type(value) is Pow:
            _, exponent = source._SCALAR_NATIVE[0](stored[0])
            if not exponent.is_integer or not 1 <= exponent.exact_numerator() <= 64:
                raise NotImplementedError("triangular polynomial powers require exponents from 1 to 64")
        elif type(value) is source._IMAGINARY_TYPE:
            raise NotImplementedError("symbolic triangular spectrum requires rational polynomial coefficients")

    def unchanged():
        if _active_callback_session() is not session:
            raise RuntimeError("symbolic spectrum request changed during evaluation")
        symbols._check_coordinates(records)
        source._unchanged(matrix, base, state, retained, scalars)

    unchanged()
    representatives = {}
    multiplicities = {}
    cache = {}
    for i in range(n):
        value = cells.get((i, i), _ZERO)
        owner = id(value)
        if owner not in cache:
            raw = source._SCALAR_SLOTS[0].__get__(value, source.Basic)
            cache[owner] = _polynomial_key(raw, session)
        key = cache[owner]
        if not key:
            root = _ZERO
        elif len(key) == 1 and not key[0][0]:
            root = Rational(key[0][1], key[0][2])
        else:
            root = representatives.setdefault(key, value)
        multiplicities[root] = multiplicities.get(root, 0) + 1
    _charge_nested_if_active(1 + n)
    unchanged()
    return multiplicities
