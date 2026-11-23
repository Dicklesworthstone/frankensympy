"""Source-bound exact dense pseudoinverses and least squares over Q(i).

The complex lanes share the original Gaussian scalar admission used by matrix
definiteness. Native results are checked against the original complex sources,
using Hermitian identities rather than a transpose-only normal equation.
The bounded complex lanes and active callback requests support at most 32 rows
and columns. Ordinary literal-rational calls retain their existing bounds.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _definiteness as _source


_NATIVE = _source._NATIVE[_source.Matrix]
_NAMES = ("shape", "is_exact_rational", "validate_gaussian_pseudoinverse_source",
          "pinv", "solve_least_squares", "least_squares_family", "to_sparse", "gaussian_qr")
_METHODS = tuple(_source._DICT.__get__(_NATIVE)[name] for name in _NAMES)
_CLASSES = (_source.Matrix, _source.ImmutableDenseMatrix)
_SPARSE_CLASSES = (_source.SparseMatrix, _source.ImmutableSparseMatrix)
_SPARSE_NATIVE = _source._NATIVE[_source.SparseMatrix]
_SPARSE_NAMES = ("shape", "is_exact_rational", "validate_gaussian_pseudoinverse_source", "to_dense")
_SPARSE_METHODS = tuple(_source._DICT.__get__(_SPARSE_NATIVE)[name] for name in _SPARSE_NAMES)


def _interfaces(operation, context, base):
    _source._interfaces()
    namespace = _source._DICT.__get__(_NATIVE)
    required = (0, 1, 2, operation, 6) if operation == 5 else (0, 1, 2, operation)
    if any(namespace.get(_NAMES[index]) is not _METHODS[index]
           for index in required):
        raise NotImplementedError("%s requires original native interfaces" % context)
    if base is _source.SparseMatrix:
        namespace = _source._DICT.__get__(_SPARSE_NATIVE)
        if any(namespace.get(name) is not method
               for name, method in zip(_SPARSE_NAMES, _SPARSE_METHODS)):
            raise NotImplementedError("%s requires original sparse native interfaces" % context)


def _needs_gaussian(matrix):
    """Inspect original borrowed storage to select one lane for both inputs."""
    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        return True
    base = _source.Matrix if dense else _source.SparseMatrix
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    slots = _source._SLOTS[base]
    native = slots[0].__get__(matrix, base)
    surface = slots[2].__get__(matrix, base)
    if type(native) is not native_type or (
            surface is not None and type(surface) is not (tuple if dense else dict)):
        return True
    cells = surface if dense or surface is None else dict.values(surface)
    return not methods[1].__get__(native, native_type) or (
        surface is not None and any(
            not any(type(entry) is atom for atom in _source._ATOMS) for entry in cells))


def _capture(matrix, session, operation, context, *, allow_sparse=False, force_gaussian=False):
    """Bind one original source and retain its publication-time check."""
    _charge_nested_if_active()
    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = allow_sparse and any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        raise NotImplementedError("%s requires a built-in dense Matrix" % context)
    base = _source.Matrix if dense else _source.SparseMatrix
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    _interfaces(operation, context, base)
    slots = _source._SLOTS[base]
    state = tuple(slot.__get__(matrix, base) for slot in slots)
    native, shape, surface, generation = state
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("%s has an invalid source snapshot" % context)
    if (type(native) is not native_type
            or methods[0].__get__(native, native_type) != shape):
        raise ValueError("%s requires intact native matrix storage" % context)
    rows, columns = shape
    storage_type = tuple if dense else dict
    if surface is not None and (type(surface) is not storage_type
            or (len(surface) != rows*columns if dense else len(surface) > rows*columns)):
        raise ValueError("%s retained cells do not match its shape" % context)
    if sparse and surface is not None and any(
            type(key) is not tuple or len(key) != 2
            or any(type(index) is not int for index in key)
            or not (0 <= key[0] < rows and 0 <= key[1] < columns)
            for key in surface):
        raise ValueError("%s has an invalid retained sparse coordinate" % context)

    # Select from the original source grammar. A held rational value may still
    # contain an undefined pole and cannot enter the literal-rational shortcut.
    preflighted = session is not None or force_gaussian
    if preflighted:
        methods[2](native, session=session)
    cells = surface if dense or surface is None else dict.values(surface)
    literal = methods[1].__get__(native, native_type) and (
        surface is None or all(any(type(entry) is atom for atom in _source._ATOMS)
                               for entry in cells)
    )
    rational_snapshot = None
    scalars = ()
    retained = surface
    if literal:
        _charge_nested_if_active(rows*columns + 1)
        captured = _source._rational_source_snapshot(matrix, dense=dense, context=context)
        if captured is None:
            raise RuntimeError("%s lost its original rational source" % context)
        native, rational_snapshot = captured
        if sparse:
            retained = rational_snapshot[7]
    else:
        # Bound the complete original native graph before copying its cells.
        if not preflighted:
            methods[2](native, session=session)
        if surface is not None:
            retained = surface if dense else dict(surface)
            scalars = _source._closed(
                retained if dense else dict.values(retained), gaussian=True)
            if scalars is None:
                raise NotImplementedError(
                    "%s requires closed exact Gaussian-rational cells" % context)
            _source._bind_gaussian(native, base, retained, scalars)

    def unchanged():
        _interfaces(operation, context, base)
        if _active_callback_session() is not session:
            raise RuntimeError("%s request changed during evaluation" % context)
        if rational_snapshot is not None:
            _source._check_rational_source(rational_snapshot)
        _source._unchanged(matrix, base, state, retained, scalars)

    if sparse:
        unchanged()
        _charge_nested_if_active(rows*columns + 1)
        converted = _SPARSE_METHODS[3](native)
        unchanged()
        if (type(converted) is not _NATIVE
                or _METHODS[0].__get__(converted, _NATIVE) != shape):
            raise RuntimeError("%s returned an invalid dense RHS snapshot" % context)
        return converted, shape, unchanged
    return native, shape, unchanged


def _publish(native, shape, cls, unchanged, context):
    if (type(native) is not _NATIVE
            or _METHODS[0].__get__(native, _NATIVE) != shape):
        raise RuntimeError("%s returned invalid output storage or shape" % context)
    _charge_nested_if_active()
    # Move accepted native storage through original slots, without invoking
    # replaced constructors or _new callbacks during result publication.
    result = object.__new__(cls)
    for slot, value in zip(_source._SLOTS[_source.Matrix], (native, shape, None, 0)):
        slot.__set__(result, value)
    unchanged()
    _charge_nested_if_active()
    return result


def pseudoinverse(matrix):
    """Return a pseudoinverse only for the retained original matrix."""
    session = _active_callback_session()
    native, (rows, columns), unchanged = _capture(matrix, session, 3, "pseudoinverse")
    unchanged()
    inverse = _METHODS[3](native, session=session)
    unchanged()
    return _publish(inverse, (columns, rows), type(matrix), unchanged, "pseudoinverse")


def solve_least_squares(matrix, rhs):
    """Return the unique Hermitian least-squares solution for original A/B."""
    _charge_nested_if_active()
    session = _active_callback_session()
    gaussian = session is not None or _needs_gaussian(matrix) or _needs_gaussian(rhs)
    left, (_, columns), left_unchanged = _capture(
        matrix, session, 4, "least squares", force_gaussian=gaussian)
    right, (_, rhs_columns), right_unchanged = _capture(
        rhs, session, 4, "least squares", allow_sparse=True, force_gaussian=gaussian)

    def unchanged():
        left_unchanged()
        right_unchanged()

    unchanged()
    solution = _METHODS[4](left, right, session=session)
    unchanged()
    return _publish(solution, (columns, rhs_columns), type(matrix), unchanged, "least squares")


def pinv_solve(matrix, rhs, arbitrary_matrix=None):
    """Return the complete checked Q(i) family X + P*W for original A/B.

    X is the unique minimum-norm least-squares solution; P is the Hermitian
    projector onto the complete kernel of A. Both are independently checked
    natively before the common bounded parameter renderer sees them.
    """
    from ._sparse_family import _ParameterMatrix, _render_family

    if not any(type(matrix) is cls for cls in _CLASSES):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    if not any(type(rhs) is cls for cls in _CLASSES + _SPARSE_CLASSES):
        raise TypeError("pinv_solve requires a built-in Matrix right-hand side")
    _charge_nested_if_active()
    session = _active_callback_session()
    context = "least-squares family"
    # The entire family lane, including literal rational sources, has the same
    # 32-axis bound. Admit both originals before sparse conversion or solving.
    left, (rows, columns), left_unchanged = _capture(
        matrix, session, 5, context, force_gaussian=True)
    right, (rhs_rows, rhs_columns), right_unchanged = _capture(
        rhs, session, 5, context, allow_sparse=True, force_gaussian=True)
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")

    def unchanged():
        left_unchanged()
        right_unchanged()

    unchanged()
    shape = (columns, rhs_columns)
    supplied = (None if arbitrary_matrix is None else
                _ParameterMatrix(arbitrary_matrix, shape, session=session))
    if supplied is not None:
        _charge_nested_if_active(sum(supplied.sizes.values()) + 1)
    kinds = (type(matrix), type(rhs), type(arbitrary_matrix))
    result_type = (_source.ImmutableSparseMatrix if _source.ImmutableSparseMatrix in kinds else
                   _source.ImmutableDenseMatrix if _source.ImmutableDenseMatrix in kinds else
                   _source.Matrix)
    unchanged()
    minimum, projector = _METHODS[5](left, right, session=session)
    unchanged()
    if (type(minimum) is not _NATIVE or type(projector) is not _NATIVE
            or _METHODS[0].__get__(minimum, _NATIVE) != shape
            or _METHODS[0].__get__(projector, _NATIVE) != (columns, columns)):
        raise RuntimeError("native least-squares family has invalid output shapes")
    _charge_nested_if_active(columns * (columns + rhs_columns) + 1)
    minimum = _METHODS[6](minimum)
    projector = _METHODS[6](projector)
    unchanged()
    return _render_family(minimum, projector, shape, result_type, supplied, unchanged)
