"""Source-bound exact dense pseudoinverses over the rationals and Q(i).

The complex lane shares the original Gaussian scalar admission used by matrix
definiteness. Its native result independently satisfies all four Hermitian
Penrose identities; no numerical rank estimate or transpose-only check is used.
The bounded complex lane and active callback requests support at most 32 rows
and columns. The ordinary literal-rational lane retains its existing bounds.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import _definiteness as _source


_NATIVE = _source._NATIVE[_source.Matrix]
_NAMES = ("shape", "is_exact_rational", "validate_gaussian_pseudoinverse_source", "pinv")
_METHODS = tuple(_source._DICT.__get__(_NATIVE)[name] for name in _NAMES)
_CLASSES = (_source.Matrix, _source.ImmutableDenseMatrix)


def _interfaces():
    _source._interfaces()
    namespace = _source._DICT.__get__(_NATIVE)
    if any(namespace.get(name) is not original
           for name, original in zip(_NAMES, _METHODS)):
        raise NotImplementedError("pseudoinverse requires original native interfaces")


def pseudoinverse(matrix):
    """Return an exact pseudoinverse only for the retained original matrix."""
    _charge_nested_if_active()
    _interfaces()
    if not any(type(matrix) is cls for cls in _CLASSES):
        raise NotImplementedError("pseudoinverse requires a built-in dense Matrix")
    session = _active_callback_session()
    base = _source.Matrix
    slots = _source._SLOTS[base]
    state = tuple(slot.__get__(matrix, base) for slot in slots)
    native, shape, surface, generation = state
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("pseudoinverse has an invalid source snapshot")
    if (type(native) is not _NATIVE
            or _METHODS[0].__get__(native, _NATIVE) != shape):
        raise ValueError("pseudoinverse requires intact native matrix storage")
    rows, columns = shape
    if surface is not None and (type(surface) is not tuple or len(surface) != rows*columns):
        raise ValueError("pseudoinverse retained cells do not match its shape")

    # Choose from the original source grammar before mathematical work. A held
    # expression may evaluate to a rational while still containing a pole; it
    # must not enter the literal-rational shortcut merely because of its value.
    literal = _METHODS[1].__get__(native, _NATIVE) and (
        surface is None or all(any(type(entry) is atom for atom in _source._ATOMS)
                               for entry in surface)
    )
    rational_snapshot = None
    scalars = ()
    if literal:
        _charge_nested_if_active(rows*columns + 1)
        captured = _source._rational_source_snapshot(
            matrix, dense=True, context="pseudoinverse",
        )
        if captured is None:
            raise RuntimeError("pseudoinverse lost its original rational source")
        native, rational_snapshot = captured
    else:
        # Borrowed native admission bounds dimensions, coefficient heights and
        # the complete original graph before Python copies any native cells.
        _METHODS[2](native, session=session)
        if surface is not None:
            scalars = _source._closed(surface, gaussian=True)
            if scalars is None:
                raise NotImplementedError("pseudoinverse requires closed exact Gaussian-rational cells")
            _source._bind_gaussian(native, base, surface, scalars)

    def unchanged():
        _interfaces()
        if _active_callback_session() is not session:
            raise RuntimeError("pseudoinverse request changed during evaluation")
        if rational_snapshot is not None:
            _source._check_rational_source(rational_snapshot)
        _source._unchanged(matrix, base, state, surface, scalars)

    unchanged()
    inverse = _METHODS[3](native, session=session)
    unchanged()
    output_shape = (columns, rows)
    if (type(inverse) is not _NATIVE
            or _METHODS[0].__get__(inverse, _NATIVE) != output_shape):
        raise RuntimeError("pseudoinverse returned invalid output storage or shape")
    _charge_nested_if_active()
    # Publication moves the accepted immutable handle through original slots.
    # A replaced constructor or _new method cannot mutate inputs or substitute
    # a different mathematical result during lifting.
    result = object.__new__(type(matrix))
    for slot, value in zip(slots, (inverse, output_shape, None, 0)):
        slot.__set__(result, value)
    unchanged()
    _charge_nested_if_active()
    return result
