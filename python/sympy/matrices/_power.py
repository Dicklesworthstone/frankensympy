"""Original-source integer powers for bounded exact dense and sparse matrices.

The native Q(i) lane checks the proposed power with an independent replay.
Negative exponents also require a two-sided inverse. Sparse sources are bounded
before a dense view is allocated; this is not a large sparse power algorithm.
Standalone dense symbolic and large literal-rational powers retain their
existing preselected implementation and make no Gaussian checker claim.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _pseudoinverse as _exact


_CONTEXT = "exact matrix integer power"
_OPERATION = 20
_SOURCE = _exact._source
_NUMERATOR = _SOURCE._DICT.__get__(_SOURCE._NATIVE_EXPR)["exact_numerator"]
_INTEGER_TYPES = (_SOURCE.Integer, _SOURCE.Zero, _SOURCE.One, _SOURCE.NegativeOne)


def _exponent_interfaces():
    _SOURCE._interfaces()
    if (_SOURCE._DICT.__get__(_SOURCE._NATIVE_EXPR).get("exact_numerator")
            is not _NUMERATOR):
        raise NotImplementedError("matrix power requires original native integer interfaces")


class _Exponent:
    """Retain a literal exponent without invoking coercion or scalar hooks."""

    def __init__(self, exponent):
        self.original = exponent
        self.session = _active_callback_session()
        self.state = None
        if type(exponent) is int:
            value = exponent
        elif any(type(exponent) is cls for cls in _INTEGER_TYPES):
            self.state = tuple(_SOURCE._slot(slot, exponent)
                               for slot in _SOURCE._SCALAR_SLOTS)
            native = self.state[0]
            if (type(native) is not _SOURCE._NATIVE_EXPR
                    or any(item is not None and (type(item) is not tuple or item)
                           for item in self.state[1:])):
                raise ValueError("matrix power exponent must be an intact literal integer")
            # Bound the native payload before copying its exact numerator.
            size = _SOURCE._SCALAR_NATIVE[-1](native)
            _charge_nested_if_active(size + 1)
            if not _SOURCE._SCALAR_NATIVE[1].__get__(native, _SOURCE._NATIVE_EXPR):
                raise ValueError("matrix power exponent payload is not an integer")
            fixed = _SOURCE._FIXED_NATIVE.get(type(exponent))
            if fixed is not None and _SOURCE._NATIVE_EQUAL(native, fixed) is not True:
                raise ValueError("matrix power exponent constant has inconsistent storage")
            value = _NUMERATOR(native)
        else:
            raise TypeError("Matrix power only supports literal integers")
        if type(value) is not int or int.bit_length(value) > 64:
            raise ValueError("matrix power exceeds its exponent limit")
        self.value = value

    def unchanged(self):
        _exponent_interfaces()
        if _active_callback_session() is not self.session:
            raise RuntimeError("matrix power request changed during evaluation")
        if self.state is not None and any(
                _SOURCE._slot(slot, self.original) is not original
                for slot, original in zip(_SOURCE._SCALAR_SLOTS, self.state)):
            raise RuntimeError("matrix power exponent mutated during evaluation")


def _legacy_dense_power(matrix, exponent):
    """The existing standalone dense lane, selected before exact arithmetic."""
    from .dense import _MAX_MATRIX_MULTIPLICATION_OPS, eye

    if matrix._native_storage is None and exponent >= 0:
        if not matrix.is_square:
            raise ValueError(f"Matrix must be square: shape is {matrix.rows}x{matrix.cols}")
        products = exponent.bit_count() + max(0, exponent.bit_length() - 1)
        if products * matrix.rows ** 3 > _MAX_MATRIX_MULTIPLICATION_OPS:
            raise ValueError("shell matrix power exceeds its aggregate operation limit")
        result, factor = matrix._new(eye(matrix.rows)), matrix
        while exponent:
            if exponent & 1:
                result = result @ factor
            exponent >>= 1
            if exponent:
                factor = factor @ factor
        return result
    if exponent < 0:
        return matrix._new(matrix.inv()._native ** (-exponent))
    return matrix._new(matrix._native ** exponent)


def power(matrix, exponent):
    """Compute A**n, retaining the coefficient class and every original owner."""
    _charge_nested_if_active()
    _exponent_interfaces()
    exponent = _Exponent(exponent)
    session = exponent.session
    original = matrix
    matrix, owner_unchanged = _exact._legacy_rational_view(matrix, _OPERATION, _CONTEXT)
    dense = any(parent is _SOURCE.Matrix for parent in _SOURCE._MRO.__get__(type(matrix)))
    if not dense and matrix is not original:
        raise NotImplementedError("custom sparse matrix subclasses require a supervised override lane")
    base = _SOURCE.Matrix if dense else _SOURCE.SparseMatrix
    native_type = _exact._NATIVE if dense else _exact._SPARSE_NATIVE
    methods = _exact._METHODS if dense else _exact._SPARSE_METHODS
    _exact._interfaces(_OPERATION, _CONTEXT, base)
    state = tuple(slot.__get__(matrix, base) for slot in _SOURCE._SLOTS[base])
    stored, stored_shape, surface, generation = state
    literal = False
    legacy = stored is None and dense and session is None
    if not legacy:
        if type(stored) is not native_type:
            raise NotImplementedError("matrix power requires closed native matrix storage")
        shape = methods[0].__get__(stored, native_type)
        if (type(stored_shape) is not tuple or len(stored_shape) != 2
                or any(type(size) is not int or size < 0 for size in stored_shape)
                or stored_shape != shape or type(generation) is not int or generation < 0):
            raise ValueError("matrix power has an invalid source snapshot")
        literal = not _exact._needs_gaussian(matrix)
        legacy = session is None and dense and literal and max(shape) > 32
        if session is None and not literal:
            syntax = _exact._METHODS[9] if dense else _exact._SPARSE_METHODS[4]
            selected = syntax(stored, session=None)
            owner_unchanged()
            _SOURCE._unchanged(matrix, base, state, surface)
            if not selected:
                if not dense:
                    raise NotImplementedError("sparse integer powers require closed exact Q(i) cells")
                legacy = True

    source_unchanged = lambda: _SOURCE._unchanged(matrix, base, state, surface)
    if not legacy or literal:
        native, shape, source_unchanged = _exact._capture(
            matrix, session, _OPERATION, _CONTEXT,
            allow_sparse=True, force_gaussian=not legacy)

    def unchanged():
        owner_unchanged()
        source_unchanged()
        exponent.unchanged()

    unchanged()
    if legacy:
        result = _legacy_dense_power(original, exponent.value)
    else:
        if not -(1 << 63) <= exponent.value < (1 << 63):
            raise ValueError("checked matrix power exponent must fit signed 64 bits")
        result = _exact._METHODS[_OPERATION](native, exponent.value, session=session)
        unchanged()
        if exponent.value == 1:
            # SymPy preserves identity here, after original-source admission.
            result = original
        elif matrix is not original:
            result = original._new(result)
        else:
            from ._row_reduction import _publish

            result = _publish(result, shape, type(original), unchanged, _CONTEXT)
    unchanged()
    _charge_nested_if_active()
    return result
