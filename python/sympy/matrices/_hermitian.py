"""Source-bound positive-definite Hermitian decompositions over exact Q(i).

The native lane verifies A = L*D*L.H, positive rational pivots and the
positive-root normalization of every Cholesky column. Original storage,
held scalar children, class owners and active requests survive publication.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _pseudoinverse as _exact
from ._row_reduction import _publish


def decomposition_if_needed(matrix, *, cholesky=False, hermitian=True):
    """Select the checked Hermitian lane before any decomposition arithmetic.

    None selects an existing standalone dense symbolic/transpose Cholesky
    lane. Large literal-rational dense LDL retains its previous native
    envelope, with positive pivots checked before a Hermitian publication.
    """
    if type(hermitian) is not bool:
        raise TypeError("hermitian must be a bool")
    if not hermitian:
        if _active_callback_session() is not None:
            _charge_nested_if_active()
            raise NotImplementedError(
                "transpose decompositions under an active request require a supervised lane")
        return None
    context = "Hermitian Cholesky" if cholesky else "Hermitian LDL"
    operation = 18 if cholesky else 17
    original = matrix
    matrix, owner_unchanged = _exact._legacy_rational_view(matrix, operation, context)
    source = _exact._source
    parents = source._MRO.__get__(type(matrix))
    dense = any(parent is source.Matrix for parent in parents)
    base = source.Matrix if dense else source.SparseMatrix
    native_type = _exact._NATIVE if dense else _exact._SPARSE_NATIVE
    methods = _exact._METHODS if dense else _exact._SPARSE_METHODS
    _exact._interfaces(operation, context, base)
    state = tuple(slot.__get__(matrix, base) for slot in source._SLOTS[base])
    stored, stored_shape, surface, generation = state
    session = _active_callback_session()
    if stored is None and dense and session is None:
        owner_unchanged()
        source._unchanged(matrix, base, state, surface)
        return None
    if type(stored) is not native_type:
        raise NotImplementedError("%s requires closed native matrix storage" % context)
    shape = methods[0].__get__(stored, native_type)
    if (type(stored_shape) is not tuple or len(stored_shape) != 2
            or any(type(size) is not int or size < 0 for size in stored_shape)
            or stored_shape != shape or type(generation) is not int or generation < 0):
        raise ValueError("%s has an invalid source snapshot" % context)
    literal = not _exact._needs_gaussian(matrix)
    legacy = session is None and dense and literal and max(shape) > 32
    if legacy and cholesky:
        # Preserve the established large literal-rational dense recurrence.
        # It does not claim the bounded Gaussian receipt or active metering.
        owner_unchanged()
        source._unchanged(matrix, base, state, surface)
        return None
    if session is None and not literal:
        syntax = _exact._METHODS[9] if dense else _exact._SPARSE_METHODS[4]
        selected = syntax(stored, session=None)
        owner_unchanged()
        source._unchanged(matrix, base, state, surface)
        if not selected:
            return None
    native_operation = 19 if legacy else operation
    native, shape, source_unchanged = _exact._capture(
        matrix, session, native_operation, context,
        allow_sparse=True, force_gaussian=not legacy)

    def unchanged():
        owner_unchanged()
        source_unchanged()

    unchanged()
    if cholesky:
        factors = (_exact._METHODS[18](native, session=session),)
    elif legacy:
        factors = _exact._METHODS[19](native, positive=True)
    else:
        factors = _exact._METHODS[17](native, session=session)
    unchanged()
    expected_count = 1 if cholesky else 2
    if type(factors) is not tuple or len(factors) != expected_count:
        raise RuntimeError("%s returned an invalid factor family" % context)

    def publish(factor):
        if (type(factor) is not _exact._NATIVE
                or _exact._METHODS[0].__get__(factor, _exact._NATIVE) != shape):
            raise RuntimeError("%s returned inconsistent factor storage" % context)
        if matrix is not original:
            unchanged()
            result = original._new(factor)
            unchanged()
            return result
        return _publish(factor, shape, type(matrix), unchanged, context)

    result = tuple(publish(factor) for factor in factors)
    unchanged()
    _charge_nested_if_active()
    return result[0] if cholesky else result
