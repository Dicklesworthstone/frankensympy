"""Source-bound exact thin QR over bounded Gaussian-rational entries.

Native modified Gram-Schmidt only proposes rational orthogonal columns.
Independent reconstruction, Hermitian Gram and pivot-minor checks establish
the complete rank. Positive exact square roots normalize the accepted factors.
Both returned objects are private until source and request checks complete.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from ._pseudoinverse import _METHODS, _NATIVE, _capture
from ._row_reduction import _publish


def gaussian_qr(matrix, *, normalize=True):
    """Return both checked factors with the original dense or sparse class."""
    if type(normalize) is not bool:
        raise TypeError("QR normalization must be a bool")
    session = _active_callback_session()
    native, (rows, columns), unchanged = _capture(
        matrix, session, 10, "QR decomposition", allow_sparse=True, force_gaussian=True)
    unchanged()
    q, r = _METHODS[10](native, normalize=normalize, session=session)
    unchanged()
    if type(q) is not _NATIVE or type(r) is not _NATIVE:
        raise RuntimeError("QR decomposition returned invalid native factors")
    q_shape = _METHODS[0].__get__(q, _NATIVE)
    r_shape = _METHODS[0].__get__(r, _NATIVE)
    rank = q_shape[1]
    if q_shape != (rows, rank) or r_shape != (rank, columns) or rank > min(rows, columns):
        raise RuntimeError("QR decomposition returned inconsistent factor shapes")
    q = _publish(q, q_shape, type(matrix), unchanged, "QR decomposition")
    r = _publish(r, r_shape, type(matrix), unchanged, "QR decomposition")
    unchanged()
    _charge_nested_if_active()
    return q, r
