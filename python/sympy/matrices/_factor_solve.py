"""Source-bound LDL/Cholesky solve entry points over exact Q(i).

Both public methods solve square systems and minimize the Hermitian residual
of tall systems. The existing native kernels prove uniqueness before returning
any RHS, including an empty one. Literal rational sparse coefficients retain
their sparse kernel; complex sources and active requests use the bounded
Gaussian checker after admission of both original operands.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _pseudoinverse as _exact
from . import sparse as _sparse


_CONTEXT = "Hermitian factor solve"


def _shape(rows, columns, rhs_rows):
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")
    if rows < columns:
        raise NotImplementedError(
            "Under-determined System. Try M.gauss_jordan_solve(rhs)")


def _rational_sparse(matrix, rhs, owners_unchanged):
    """Keep arbitrary admitted sparse dimensions out of dense storage."""
    rhs_sparse = any(type(rhs) is cls for cls in _exact._SPARSE_CLASSES)
    left = _sparse._rational_source_snapshot(matrix, context=_CONTEXT)
    right = _sparse._rational_source_snapshot(rhs, dense=not rhs_sparse, context=_CONTEXT)
    if left is None or right is None:
        raise RuntimeError("Hermitian factor solve lost its original rational sources")
    coefficients, source_snapshot = left
    rhs_native, rhs_snapshot = right
    rows, columns = source_snapshot[4]
    rhs_rows, rhs_columns = rhs_snapshot[4]
    _shape(rows, columns, rhs_rows)

    def unchanged():
        owners_unchanged()
        _exact._interfaces(4, _CONTEXT, _exact._source.SparseMatrix)
        if _active_callback_session() is not None:
            raise RuntimeError("Hermitian factor solve request changed during evaluation")
        _sparse._check_rational_source(source_snapshot)
        _sparse._check_rational_source(rhs_snapshot)

    unchanged()
    if not rhs_sparse:
        rhs_native = _sparse._SOLVE_DENSE_MEMBERS["to_sparse"](rhs_native)
    unchanged()
    solution = _sparse._REDUCTION_NATIVE_MEMBERS["solve_least_squares"](
        coefficients, rhs_native)
    unchanged()
    if (type(solution) is not _sparse._NativeSparseMatrix
            or _sparse._REDUCTION_NATIVE_MEMBERS["shape"].__get__(
                solution, _sparse._NativeSparseMatrix) != (columns, rhs_columns)):
        raise RuntimeError("Hermitian factor solve returned invalid sparse output")
    # Unlike mutable SparseMatrix.solve_least_squares, these public methods
    # retain the coefficient class independently of the RHS class.
    result = _sparse._checked_sparse_lift(solution, type(matrix))
    unchanged()
    return result


def solve(matrix, rhs):
    """Return the unique least-squares solution in the coefficient class."""
    source = _exact._source
    if not any(parent is source.Matrix or parent is source.SparseMatrix
               for parent in source._MRO.__get__(type(rhs))):
        raise TypeError("factor solve requires a Matrix right-hand side")
    _charge_nested_if_active()
    session = _active_callback_session()
    original_matrix, original_rhs = matrix, rhs
    # Dense LDLsolve already accepted ordinary inherited rational subclasses.
    # Keep that preselected lane and its construction hook, with the same
    # original-owner rechecks used by exact solve. No symbolic fallback occurs.
    matrix, matrix_owner_unchanged = _exact._legacy_rational_view(matrix, 4, _CONTEXT)
    rhs, rhs_owner_unchanged = _exact._legacy_rational_view(rhs, 4, _CONTEXT)
    sparse = any(type(matrix) is cls for cls in _exact._SPARSE_CLASSES)
    if sparse and matrix is not original_matrix:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    gaussian = session is not None or _exact._needs_gaussian(matrix) or _exact._needs_gaussian(rhs)
    if gaussian and (matrix is not original_matrix or rhs is not original_rhs):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")

    def owners_unchanged():
        matrix_owner_unchanged()
        rhs_owner_unchanged()

    if sparse and not gaussian:
        return _rational_sparse(matrix, rhs, owners_unchanged)

    left, (rows, columns), left_unchanged = _exact._capture(
        matrix, session, 4, _CONTEXT, allow_sparse=True, force_gaussian=gaussian)
    right, (rhs_rows, rhs_columns), right_unchanged = _exact._capture(
        rhs, session, 4, _CONTEXT, allow_sparse=True, force_gaussian=gaussian)
    _shape(rows, columns, rhs_rows)

    def unchanged():
        owners_unchanged()
        left_unchanged()
        right_unchanged()

    unchanged()
    solution = _exact._METHODS[4](left, right, session=session)
    unchanged()
    shape = (columns, rhs_columns)
    if (type(solution) is not _exact._NATIVE
            or _exact._METHODS[0].__get__(solution, _exact._NATIVE) != shape):
        raise RuntimeError("Hermitian factor solve returned invalid dense output")
    cls = type(original_matrix)
    if not any(cls is builtin for builtin in _exact._CLASSES + _exact._SPARSE_CLASSES):
        result = original_matrix._new(solution)
        unchanged()
        _charge_nested_if_active()
        return result
    from ._row_reduction import _publish

    return _publish(solution, shape, cls, unchanged, _CONTEXT)
