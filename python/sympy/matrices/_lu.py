"""Source-bound rectangular LU with exact factors and ordered row swaps.

The native checker establishes P*A=L*U, a unit lower-triangular L, the
complete echelon pivots of U, and the permutation induced by every swap.
The bounded Q(i) lane never falls back after admission. Existing standalone
literal-rational square decompositions retain their larger native envelope.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _pseudoinverse as _exact
from ._row_reduction import _publish


def _legacy_swaps(permutation, rows):
    """Express an independently checked permutation as forward row swaps."""
    source = _exact._source
    values = source._NATIVE_CELLS[source.Matrix][1](permutation)
    one = source._FIXED_NATIVE[source.One]
    ordering = []
    for row in range(rows):
        positions = [column for column in range(rows)
                     if source._NATIVE_EQUAL(values[row*rows + column], one)]
        if len(positions) != 1:
            raise RuntimeError("LU decomposition returned an invalid permutation")
        ordering.append(positions[0])
    if len(set(ordering)) != rows:
        raise RuntimeError("LU decomposition returned an invalid permutation")
    current, swaps = list(range(rows)), []
    for row, desired in enumerate(ordering):
        if current[row] != desired:
            other = current.index(desired, row + 1)
            current[row], current[other] = current[other], current[row]
            swaps.append([row, other])
    return swaps


def lu_decomposition(matrix, *, iszerofunc=None, simpfunc=None,
                     rankcheck=False, permutation_matrix=False):
    if iszerofunc is not None or simpfunc is not None:
        raise NotImplementedError(
            "exact LU decomposition with custom callbacks requires a supervised override lane")
    if type(rankcheck) is not bool:
        raise TypeError("LU rankcheck must be a bool")
    session = _active_callback_session()
    original = matrix
    matrix, owner_unchanged = _exact._legacy_rational_view(matrix, 14, "LU decomposition")
    dense = any(type(matrix) is cls for cls in _exact._CLASSES)
    base = _exact._source.Matrix if dense else _exact._source.SparseMatrix
    _exact._interfaces(14, "LU decomposition", base)
    stored = _exact._source._SLOTS[base][0].__get__(matrix, base)
    native_type = _exact._NATIVE if dense else _exact._SPARSE_NATIVE
    methods = _exact._METHODS if dense else _exact._SPARSE_METHODS
    if type(stored) is not native_type:
        raise NotImplementedError("LU decomposition requires closed exact matrix entries")
    rows, columns = methods[0].__get__(stored, native_type)
    legacy = (session is None and rows == columns and not rankcheck
              and (max(rows, columns) > 32 or matrix is not original)
              and not _exact._needs_gaussian(matrix))
    if matrix is not original and not legacy:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    native, shape, source_unchanged = _exact._capture(
        matrix, session, 15 if legacy else 14, "LU decomposition",
        allow_sparse=True, force_gaussian=not legacy)

    def unchanged():
        owner_unchanged()
        source_unchanged()

    unchanged()
    if legacy:
        p, l, u = _exact._METHODS[15](native)
        unchanged()
        swaps = _legacy_swaps(p, rows)
    else:
        p, l, u, raw_swaps, pivots = _exact._METHODS[14](native, session=session)
        unchanged()
        if (type(raw_swaps) is not list or len(raw_swaps) > min(shape)
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(index) is not int or not 0 <= index < rows
                              for index in pair) for pair in raw_swaps)
                or type(pivots) is not list or len(pivots) > min(shape)
                or any(type(column) is not int or not 0 <= column < columns
                       for column in pivots)
                or any(left >= right for left, right in zip(pivots, pivots[1:]))):
            raise RuntimeError("LU decomposition returned invalid swaps or pivots")
        if rankcheck and len(pivots) < min(shape):
            raise ValueError("Rank of matrix is strictly less than number of rows or columns")
        swaps = [list(pair) for pair in raw_swaps]
    for factor, expected in ((p, (rows, rows)), (l, (rows, rows)), (u, shape)):
        if (type(factor) is not _exact._NATIVE
                or _exact._METHODS[0].__get__(factor, _exact._NATIVE) != expected):
            raise RuntimeError("LU decomposition returned inconsistent factor shapes")

    def publish(factor, expected):
        if matrix is not original:
            # A preselected standalone literal-rational subclass keeps its
            # established constructor. It cannot enter the new complex lane.
            unchanged()
            result = original._new(factor)
            unchanged()
            return result
        return _publish(factor, expected, type(matrix), unchanged, "LU decomposition")

    l, u = publish(l, (rows, rows)), publish(u, shape)
    result = ((publish(p, (rows, rows)), l, u) if permutation_matrix else (l, u, swaps))
    unchanged()
    _charge_nested_if_active()
    return result
