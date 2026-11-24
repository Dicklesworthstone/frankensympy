"""Checked complex determinants without factorial cofactor expansion."""

from ..core import _active_callback_session, _charge_nested_if_active, _native, _wrap
from . import _pseudoinverse as _exact
from ._row_reduction import _selected


def determinant_if_needed(matrix):
    """Select the bounded exact route from the retained original source."""
    original = matrix
    matrix, owner_unchanged = _exact._legacy_rational_view(matrix, 16, "exact determinant")
    if not _selected(matrix):
        owner_unchanged()
        return None
    if matrix is not original:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    session = _active_callback_session()
    native, shape, source_unchanged = _exact._capture(
        matrix, session, 16, "exact determinant", allow_sparse=True, force_gaussian=True)

    def unchanged():
        owner_unchanged()
        source_unchanged()

    unchanged()
    value = _exact._METHODS[16](native, session=session)
    unchanged()
    if type(value) is not _native.Expr:
        raise RuntimeError("exact determinant returned invalid native output")
    _charge_nested_if_active()
    result = _wrap(value)
    if shape == (1, 1):
        dense = any(type(matrix) is cls for cls in _exact._CLASSES)
        base = _exact._source.Matrix if dense else _exact._source.SparseMatrix
        retained = _exact._source._SLOTS[base][2].__get__(matrix, base)
        if retained is not None:
            # Once the original source has passed the complete native check,
            # retain the profile's singleton held-cell identity. A hidden
            # undefined operand was already rejected and cannot escape here.
            result = retained[0] if dense else retained.get((0, 0), result)
    unchanged()
    _charge_nested_if_active()
    return result
