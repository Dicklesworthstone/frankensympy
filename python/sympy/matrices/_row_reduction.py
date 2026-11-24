"""Source-bound exact complex row reduction and complete affine families.

The native checker proves row equivalence and canonical pivot positions over
Q(i). This boundary retains the original matrix/scalar owners, shares an active
request through publication, and keeps the existing large rational sparse lane.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _pseudoinverse as _exact
from . import sparse as _sparse


def _selected(matrix):
    """Select before arithmetic; an admitted failure never tries another lane."""
    session = _active_callback_session()
    if session is not None:
        return True
    if not _exact._needs_gaussian(matrix):
        return False
    source = _exact._source
    parents = source._MRO.__get__(type(matrix))
    dense = any(parent is source.Matrix for parent in parents)
    base = source.Matrix if dense else source.SparseMatrix
    native_type = _exact._NATIVE if dense else _exact._SPARSE_NATIVE
    _exact._interfaces(11, "exact row reduction selection", base)
    state = tuple(slot.__get__(matrix, base) for slot in source._SLOTS[base])
    native, _shape, surface, _generation = state
    if native is None:
        return False
    if type(native) is not native_type:
        raise ValueError("exact row reduction requires intact native matrix storage")
    syntax = _exact._METHODS[9] if dense else _exact._SPARSE_METHODS[4]
    selected = syntax(native, session=session)
    source._unchanged(matrix, base, state, surface)
    return selected


def _publish(native, shape, cls, unchanged, context):
    if any(cls is dense for dense in _exact._CLASSES):
        return _exact._publish(native, shape, cls, unchanged, context)
    if (type(native) is not _exact._NATIVE
            or _exact._METHODS[0].__get__(native, _exact._NATIVE) != shape):
        raise RuntimeError("%s returned invalid native output" % context)
    _charge_nested_if_active(shape[0]*shape[1] + 1)
    result = _sparse._checked_sparse_lift(_exact._METHODS[6](native), cls)
    unchanged()
    _charge_nested_if_active()
    return result


def reduction_if_needed(matrix, operation, *, iszerofunc=None, simplify=False):
    """Return a checked complex result, or select the existing rational path."""
    native_operation = 12 if operation == "nullspace" else 11
    original = matrix
    matrix, owner_unchanged = _exact._legacy_rational_view(
        matrix, native_operation, "exact row reduction")
    if not _selected(matrix):
        owner_unchanged()
        return None
    if matrix is not original:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    if iszerofunc is not None or type(simplify) is not bool:
        raise NotImplementedError(
            "exact complex row reduction with custom callbacks requires a supervised override lane")
    session = _active_callback_session()
    native, shape, source_unchanged = _exact._capture(
        matrix, session, native_operation, "exact row reduction",
        allow_sparse=True, force_gaussian=True)

    def unchanged():
        owner_unchanged()
        source_unchanged()

    unchanged()
    if operation == "nullspace":
        basis = _exact._METHODS[12](native, session=session)
        unchanged()
        if type(basis) is not list or len(basis) > shape[1]:
            raise RuntimeError("exact nullspace returned an invalid basis")
        result = [_publish(vector, (shape[1], 1), type(matrix), unchanged,
                           "exact nullspace") for vector in basis]
    else:
        reduced, pivots = _exact._METHODS[11](native, session=session)
        unchanged()
        if (type(pivots) is not list or len(pivots) > min(shape)
                or any(type(column) is not int or not 0 <= column < shape[1]
                       for column in pivots)
                or any(left >= right for left, right in zip(pivots, pivots[1:]))):
            raise RuntimeError("exact row reduction returned invalid pivots")
        if operation == "rank":
            result = len(pivots)
        else:
            # The pinned rational sparse profile returns a mutable dense
            # RREF, including under an active cumulative callback session.
            cls = (_exact._source.Matrix
                   if any(type(matrix) is sparse for sparse in _exact._SPARSE_CLASSES)
                   and not _exact._needs_gaussian(matrix) else type(matrix))
            result = (_publish(reduced, shape, cls, unchanged,
                               "exact row reduction"), tuple(pivots))
    unchanged()
    _charge_nested_if_active()
    return result


def affine_if_needed(matrix, rhs, freevar):
    """Publish all exact solutions of the retained Q(i) source and RHS."""
    if not _selected(matrix) and not _selected(rhs):
        return None
    session = _active_callback_session()
    left, (rows, columns), left_unchanged = _exact._capture(
        matrix, session, 13, "exact affine solve", allow_sparse=True, force_gaussian=True)
    right, (rhs_rows, rhs_columns), right_unchanged = _exact._capture(
        rhs, session, 13, "exact affine solve", allow_sparse=True, force_gaussian=True)
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")
    if rhs_columns == 0 and columns:
        raise ValueError("exact affine solve requires a nonempty right-hand side")

    def unchanged():
        left_unchanged()
        right_unchanged()

    unchanged()
    family = _exact._METHODS[13](left, right, session=session)
    unchanged()
    if family is None:
        raise ValueError("Linear system has no solution")
    solution, parameters, free_columns = family
    if (type(free_columns) is not list or len(free_columns) > columns
            or any(type(column) is not int or not 0 <= column < columns
                   for column in free_columns)
            or any(a >= b for a, b in zip(free_columns, free_columns[1:]))):
        raise RuntimeError("exact affine solve returned invalid free columns")
    solution_shape = (columns, rhs_columns)
    parameter_shape = (len(free_columns), rhs_columns)
    for native, shape in ((solution, solution_shape), (parameters, parameter_shape)):
        if (type(native) is not _exact._NATIVE
                or _exact._METHODS[0].__get__(native, _exact._NATIVE) != shape):
            raise RuntimeError("exact affine solve returned invalid native output")
    _charge_nested_if_active(parameter_shape[0]*rhs_columns + 1)
    bindings = _sparse._affine_parameter_symbols(_exact._METHODS[6](parameters))
    solution = _publish(solution, solution_shape, type(matrix), unchanged, "exact affine solve")
    parameters = _publish(parameters, parameter_shape, type(matrix), unchanged, "exact affine solve")
    unchanged()
    _sparse._check_affine_parameters(bindings)
    _charge_nested_if_active()
    return ((solution, parameters, list(free_columns)) if freevar else (solution, parameters))


def rank_decomposition(matrix, *, iszerofunc, simplify):
    """Extract a full-rank factorization from one accepted RREF."""
    if iszerofunc is not None or type(simplify) is not bool:
        raise NotImplementedError("rank decomposition requires built-in exact zero tests")
    cls = type(matrix)
    dense = any(cls is item for item in _exact._CLASSES)
    if not dense and not any(cls is item for item in _exact._SPARSE_CLASSES):
        raise NotImplementedError("rank decomposition requires a built-in matrix")
    base = _sparse.Matrix if dense else _sparse.SparseMatrix
    slots = _exact._source._SLOTS[base]
    session = _active_callback_session()
    literal = not _exact._needs_gaussian(matrix)
    context = "exact rank decomposition"
    if literal and session is None:
        # The rational proof kernel replays row operations on the entire
        # original system. Avoid producing a dense rows-by-columns RREF
        # just to discard its trailing zero rows.
        source, snapshot = _sparse._rational_source_snapshot(matrix, dense=dense, context=context)

        def unchanged():
            _sparse._check_rational_source(snapshot)

        rows, cols = snapshot[4]
        if dense:
            source = _sparse._SOLVE_DENSE_MEMBERS["to_sparse"](source)
        reduced, pivots = _sparse._REDUCTION_NATIVE_MEMBERS["rref"](source)
    else:
        # Preserve cumulative request charging, and verify the complex row
        # transform against both sides of the original matrix.
        source, (rows, cols), unchanged = _exact._capture(
            matrix, session, 11, context, allow_sparse=True, force_gaussian=True)
        reduced, pivots = _exact._METHODS[11](source, session=session)
        _charge_nested_if_active(2*rows*cols + 1)
        source = _exact._METHODS[6](source)
        reduced = _exact._METHODS[6](reduced)
    unchanged()

    rank = len(pivots)
    right_class = _sparse.Matrix if literal else cls
    if right_class is _sparse.Matrix or dense:
        _sparse._matrix_dimensions(rank, cols)
    pivot_column = {column: index for index, column in enumerate(pivots)}
    # R[:, pivots] is [I; 0] and R is row-equivalent to A. Therefore
    # A[:, pivots] * R[:rank, :] equals A, even when rank is zero.
    stored = _sparse._REDUCTION_NATIVE_MEMBERS["todok"]
    _charge_nested_if_active(rows*rank + rank*cols + 1)  # active inputs are bounded dense views
    basis = {(r, pivot_column[c]): value for (r, c), value in stored(source).items()
             if c in pivot_column}
    left = _sparse._checked_sparse_lift(
        _sparse._NativeSparseMatrix(rows, rank, basis), cls)
    right = _sparse._checked_sparse_lift(
        _sparse._NativeSparseMatrix(rank, cols, stored(reduced)), right_class)

    # Pivot columns are a selection of original public objects. Keep their
    # owners (including held Gaussian expressions) alongside the native view.
    surface = slots[2].__get__(matrix, base)
    if surface is not None:
        if dense:
            selected = tuple(surface[row*cols + col] for row in range(rows) for col in pivots)
        else:
            selected = {(r, pivot_column[c]): value for (r, c), value in surface.items()
                        if c in pivot_column}
        slots[2].__set__(left, selected)
    unchanged()
    _charge_nested_if_active()
    return left, right
