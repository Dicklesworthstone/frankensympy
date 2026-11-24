"""Complete finite solutions of A*X == B modulo an exact integer m.

This FrankenSymPy extension returns (P, K, periods), with
X = P + K*T modulo m and independent 0 <= T[i, j] < periods[i] for every
RHS column j. The family is complete without repeated parameterizations.
Checked inconsistency returns None. A nonunique family is a successful result.

All four built-in matrix classes are supported, including mixed dense/sparse
operands. P and K preserve A's exact class; periods is a tuple of exact Integer
values. Native Smith generation, independent verification and output lifting
share the existing request and bounded integer-normal-form allowances.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _inv_mod as _modulus
from . import normalforms as _forms


_SOLVERS = {
    base: _forms._DICT.__get__(native)["solve_mod"]
    for base, native in _forms._NATIVE.items()
}


def _interfaces():
    _modulus._scalar_interfaces("solve_mod")
    for base, original in _SOLVERS.items():
        namespace = _forms._DICT.__get__(_forms._NATIVE[base])
        if namespace.get("solve_mod") is not original:
            raise NotImplementedError("solve_mod requires original native interfaces")


def _unchanged(coefficients, rhs, modulus):
    _interfaces()
    coefficients.unchanged()
    rhs.unchanged()
    modulus.unchanged()
    if (coefficients.session is not rhs.session
            or coefficients.session is not modulus.session
            or _active_callback_session() is not coefficients.session):
        raise RuntimeError("solve_mod request changed during evaluation")


def _lift_periods(values, count):
    if type(values) is not list or len(values) != count:
        raise RuntimeError("solve_mod returned invalid period storage")
    _charge_nested_if_active(count + 1)
    result = []
    for value in values:
        if (type(value) is not _forms._NATIVE_EXPR
                or _modulus._SCALAR_METHODS[0].__get__(
                    value, _forms._NATIVE_EXPR) is not True):
            raise RuntimeError("solve_mod returned a noninteger period")
        # Native verification already proves each period is in [2,m].
        # Install the exact integer directly, without user constructors,
        # string/numeric coercions, or replaceable scalar setters.
        integer = object.__new__(_forms.Integer)
        _forms._ATOM_SLOTS[0].__set__(integer, value)
        result.append(integer)
    return tuple(result)


def solve_mod(matrix, rhs, m):
    """Return the complete finite modular family (P, K, periods), or None.

    For A of shape r-by-n and B of shape r-by-k, P is n-by-k and K is n-by-s.
    Every modular solution is P + K*T, reducing entries modulo m, with each
    T[i,j] independently ranging from zero through periods[i]-1. All entries
    of P and K are canonical residues in [0,m); generators of period one
    are omitted. The number of distinct solutions is product(periods)**k.

    Integer m >= 2, composite moduli, singular/rectangular A and zero-area
    shapes are supported. Dimensions are at most 64, original entries and
    modulus at most 4096 bits. Unsupported inputs, source mutation, exhausted
    work and cancellation raise; only verified inconsistency returns None.
    A zero-column RHS has one empty solution even if periods is nonempty.
    """
    _charge_nested_if_active()
    _interfaces()
    modulus = _modulus._Modulus(m, "solve_mod")
    coefficients = _forms._IntegerSource(matrix)
    right = _forms._IntegerSource(rhs)
    _charge_nested_if_active(len(coefficients.atoms) + len(right.atoms) + 1)
    _unchanged(coefficients, right, modulus)
    for label, source in (("coefficients", coefficients), ("RHS", right)):
        for cell, state in source.atoms:
            _modulus._literal_atom(cell, state[0], label, "solve_mod")
    rows, columns = coefficients.shape
    rhs_rows, rhs_columns = right.shape
    if rows != rhs_rows:
        raise ValueError("solve_mod requires the same number of rows in A and B")

    native = _SOLVERS[coefficients.base](
        coefficients.native, right.native, modulus.native, session=coefficients.session,
    )
    _unchanged(coefficients, right, modulus)
    result = None
    if native is not None:
        if type(native) is not tuple or len(native) != 3:
            raise RuntimeError("solve_mod returned invalid result storage")
        particular, generators, native_periods = native
        base = coefficients.base
        native_type = _forms._NATIVE[base]
        if type(particular) is not native_type or type(generators) is not native_type:
            raise RuntimeError("solve_mod returned invalid native matrix owners")
        shape_descriptor = _forms._NATIVE_METHODS[base][0]
        shapes = tuple(shape_descriptor.__get__(value, native_type)
                       for value in (particular, generators))
        if (shapes[0] != (columns, rhs_columns) or shapes[1][0] != columns
                or not 0 <= shapes[1][1] <= columns):
            raise RuntimeError("solve_mod returned invalid output shapes")
        _charge_nested_if_active(sum(row * column for row, column in shapes) + 2)
        output = []
        for value, shape in zip((particular, generators), shapes):
            lifted = object.__new__(coefficients.owner)
            for slot, item in zip(_forms._SLOTS[base], (value, shape, None, 0)):
                slot.__set__(lifted, item)
            output.append(lifted)
        periods = _lift_periods(native_periods, shapes[1][1])
        result = (output[0], output[1], periods)
    # Both a finite family and None remain bound to both original matrices,
    # the original modulus and the original request through final publication.
    _charge_nested_if_active()
    _unchanged(coefficients, right, modulus)
    return result


__all__ = ["solve_mod"]
