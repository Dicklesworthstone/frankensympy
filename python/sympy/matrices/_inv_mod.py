"""Exact matrix inverses over arbitrary bounded integer residue rings.

Smith decomposition reduces the unit question to its integer invariant factors,
so composite moduli are supported even when no individual matrix entry is a
unit. Accepted output has both original modular residuals checked independently
and always uses mutable dense Matrix storage, matching SymPy's public convention.
"""

from ..core import _active_callback_session, _charge_nested_if_active
from . import normalforms as _forms
from .exceptions import NonInvertibleMatrixError, NonSquareMatrixError


_SCALAR_NAMES = ("is_integer", "is_rational", "__eq__")
_SCALAR_METHODS = tuple(_forms._DICT.__get__(_forms._NATIVE_EXPR)[name]
                        for name in _SCALAR_NAMES)
_FIXED_MODULI = {
    _forms.Zero: _forms._native.py_integer(0),
    _forms.One: _forms._native.py_integer(1),
    _forms.NegativeOne: _forms._native.py_integer(-1),
    _forms.Half: _forms._native.py_rational(1, 2),
}


# Capture interfaces at package import, before any user replacement.
_INVERSES = {
    base: _forms._DICT.__get__(native)["inverse_mod"]
    for base, native in _forms._NATIVE.items()
}


def _interfaces():
    _forms._interfaces()
    namespace = _forms._DICT.__get__(_forms._NATIVE_EXPR)
    if any(namespace.get(name) is not original
           for name, original in zip(_SCALAR_NAMES, _SCALAR_METHODS)):
        raise NotImplementedError("inv_mod requires original native scalar interfaces")
    for base, method in _INVERSES.items():
        if _forms._DICT.__get__(_forms._NATIVE[base]).get("inverse_mod") is not method:
            raise NotImplementedError("inv_mod requires original native interfaces")


def _literal_atom(value, native, label):
    integer = _SCALAR_METHODS[0].__get__(native, _forms._NATIVE_EXPR)
    rational = _SCALAR_METHODS[1].__get__(native, _forms._NATIVE_EXPR)
    if (not (integer or rational)
            or type(value) in (_forms.Integer, _forms.Zero, _forms.One,
                               _forms.NegativeOne) and not integer):
        raise ValueError("inv_mod: %s payload does not match its literal owner" % label)
    fixed = _FIXED_MODULI.get(type(value))
    if fixed is not None and _SCALAR_METHODS[2](native, fixed) is not True:
        raise ValueError("inv_mod: fixed %s constant has inconsistent native storage" % label)


class _Modulus:
    def __init__(self, value):
        self.session = _active_callback_session()
        self.original = value
        self.state = None
        if type(value) is int:
            if int.bit_length(value) > 4096:
                raise ValueError("inv_mod: modulus exceeds 4096 bits")
            self.native = value
        elif any(type(value) is atom for atom in _forms._ATOMS):
            self.state = _forms._atom_state(value)
            if type(self.state[0]) is not _forms._NATIVE_EXPR:
                raise ValueError("inv_mod: invalid native modulus storage")
            if any(item is not _forms._MISSING and item is not None
                   and (type(item) is not tuple or len(item) != 0)
                   for item in self.state[1:]):
                raise ValueError("inv_mod: modulus must be a literal integer atom")
            self.native = self.state[0]
            _literal_atom(value, self.native, "modulus")
        else:
            # Do not run __int__, __index__, sympify, or custom scalar protocols.
            raise TypeError("inv_mod: modulus m must be an integer")

    def unchanged(self):
        if _active_callback_session() is not self.session:
            raise RuntimeError("inv_mod request changed during evaluation")
        if self.state is not None and any(
            current is not original
            for current, original in zip(_forms._atom_state(self.original), self.state)
        ):
            raise RuntimeError("inv_mod original modulus mutated during evaluation")


def _unchanged(source, modulus):
    _interfaces()
    source.unchanged()
    modulus.unchanged()
    if source.session is not modulus.session:
        raise RuntimeError("inv_mod request changed during evaluation")


def inverse_mod(matrix, m):
    """Return the unique matrix inverse modulo integer m, in residues 0..m-1.

    The supported exact ring has m >= 2, at most 4096 bits in the modulus and
    original integer entries, and at most 64 rows and columns. All four
    built-in matrix classes are accepted; the result is a mutable dense Matrix.
    NonSquareMatrixError and NonInvertibleMatrixError distinguish shape and
    proven algebraic failure. Unsupported sources or exhausted/cancelled work
    raise without publishing a candidate or being labelled noninvertible.
    """
    _charge_nested_if_active()
    _interfaces()
    if not any(type(matrix) is cls for cls in _forms._CLASSES):
        raise NotImplementedError("inv_mod requires a built-in Matrix")
    base = _forms.Matrix if type(matrix) in (
        _forms.Matrix, _forms.ImmutableDenseMatrix,
    ) else _forms.SparseMatrix
    shape = _forms._SLOTS[base][1].__get__(matrix, base)
    if (type(shape) is tuple and len(shape) == 2
            and all(type(size) is int and size >= 0 for size in shape)
            and shape[0] != shape[1]):
        raise NonSquareMatrixError("Matrix must be square")
    modulus = _Modulus(m)
    try:
        source = _forms._IntegerSource(matrix)
    except _forms._NonliteralIntegerSource as error:
        raise ValueError("inv_mod: matrix entries must be integers") from error
    _charge_nested_if_active(len(source.atoms) + 1)
    for cell, state in source.atoms:
        _literal_atom(cell, state[0], "source")
    _unchanged(source, modulus)
    native = _INVERSES[source.base](
        source.native, modulus.native, session=source.session,
    )
    _unchanged(source, modulus)
    result = None
    if native is not None:
        native_type = _forms._NATIVE[_forms.Matrix]
        descriptor = _forms._NATIVE_METHODS[_forms.Matrix][0]
        if (type(native) is not native_type
                or descriptor.__get__(native, native_type) != source.shape):
            raise RuntimeError("inv_mod returned invalid output storage or shape")
        rows, columns = source.shape
        _charge_nested_if_active(rows * columns + 1)
        result = object.__new__(_forms.Matrix)
        for slot, item in zip(
            _forms._SLOTS[_forms.Matrix], (native, source.shape, None, 0),
        ):
            slot.__set__(result, item)
    _unchanged(source, modulus)
    _charge_nested_if_active()
    if result is None:
        raise NonInvertibleMatrixError(f"Matrix is not invertible (mod {m})")
    return result
