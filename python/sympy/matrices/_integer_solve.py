"""Complete checked integer matrix solution families.

This is a FrankenSymPy extension, not a SymPy compatibility claim. For
``A X = B``, ``solve_integer(A, B)`` returns ``(P, K)`` such that every integer
solution is exactly ``P + K*T`` for an arbitrary integer matrix T. It returns
``None`` only when an independently checked divisibility or zero-row
contradiction proves that the complete integer system is inconsistent.

Both operands use the original literal-integer source admission shared with
Smith normal forms. Each dimension is bounded by 64 and original coefficients
by 4096 bits; native generation, independent verification and output lifting
retain their work and coefficient limits. Dense/sparse and mutable/immutable
operands can be mixed. Both outputs preserve A's exact built-in matrix class.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import normalforms as _forms


# The package imports this module after all matrix owners have been defined,
# before user code can replace an interface and have it become the baseline.
_SOLVERS = {
    base: _forms._DICT.__get__(native)["solve_integer"]
    for base, native in _forms._NATIVE.items()
}


def _solver_interfaces():
    for base, original in _SOLVERS.items():
        namespace = _forms._DICT.__get__(_forms._NATIVE[base])
        if namespace.get("solve_integer") is not original:
            raise NotImplementedError("integer solving requires original native interfaces")


def _unchanged(coefficients, rhs):
    _solver_interfaces()
    coefficients.unchanged()
    rhs.unchanged()
    if (coefficients.session is not rhs.session
            or _active_callback_session() is not coefficients.session):
        raise RuntimeError("integer solve request changed during evaluation")


def solve_integer(matrix, rhs):
    """Return ``(particular, kernel)`` for every integer solution, or ``None``.

    For an m-by-n matrix A and an m-by-k RHS B, particular is n-by-k and
    kernel is n-by-r, where r is the integer nullity of A. Every solution is
    ``particular + kernel*T`` for an arbitrary r-by-k integer matrix T. The
    kernel spans the complete integer lattice, including primitive directions
    that separately scaled rational nullspace vectors can miss.

    A zero-column RHS is valid: particular then has zero columns, while kernel
    still describes A's entire integer nullspace. The unique empty solution
    does not make original-source validation or verification optional. All
    four built-in matrix classes are accepted; results have A's exact class.
    Unsupported sources and exhausted work raise instead of returning None.
    """
    _solver_interfaces()
    coefficients = _forms._IntegerSource(matrix)
    right = _forms._IntegerSource(rhs)
    _unchanged(coefficients, right)
    rows, columns = coefficients.shape
    rhs_rows, rhs_columns = right.shape
    if rows != rhs_rows:
        raise ValueError("integer solving requires the same number of rows in A and B")

    native = _SOLVERS[coefficients.base](
        coefficients.native, right.native, session=coefficients.session,
    )
    _unchanged(coefficients, right)
    result = None
    if native is not None:
        if type(native) is not tuple or len(native) != 2:
            raise RuntimeError("integer solve returned invalid result storage")
        base = coefficients.base
        native_type = _forms._NATIVE[base]
        if any(type(value) is not native_type for value in native):
            raise RuntimeError("integer solve returned invalid native matrix owners")
        shape_descriptor = _forms._NATIVE_METHODS[base][0]
        shapes = tuple(shape_descriptor.__get__(value, native_type) for value in native)
        if (shapes[0] != (columns, rhs_columns) or shapes[1][0] != columns
                or not 0 <= shapes[1][1] <= columns):
            raise RuntimeError("integer solve returned invalid output shapes")
        _charge_nested_if_active(sum(row * column for row, column in shapes) + 2)
        output = []
        for value, shape in zip(native, shapes):
            # Install checked native results without calling constructors,
            # replaceable setters, or conversions between public classes.
            lifted = object.__new__(type(matrix))
            for slot, item in zip(_forms._SLOTS[base], (value, shape, None, 0)):
                slot.__set__(lifted, item)
            output.append(lifted)
        result = tuple(output)
    # Inconsistency is also a mathematical result: both original sources and
    # the active request remain authoritative until that result is published.
    _unchanged(coefficients, right)
    _charge_nested_if_active()
    return result


__all__ = ["solve_integer"]
