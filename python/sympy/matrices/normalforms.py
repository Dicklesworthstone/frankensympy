"""Independently checked bounded integer Smith and column Hermite forms.

``smith_normal_form`` implements the inferred-integer slice and returns a
mutable dense Matrix, including for immutable or sparse inputs. The native
extension ``smith_normal_decomposition`` also returns integral unimodular
transforms, preserving the input's built-in matrix class. Its convention is
``D = U*A*V`` with positive nonzero invariant factors followed by zeros.

``hermite_normal_form`` returns a mutable dense canonical basis of the
integer column lattice. ``hermite_normal_decomposition`` additionally returns
V and its integral inverse with ``matrix*V == [0 | H]``, preserving the input
matrix class. Redundant columns are removed; every original row is retained.

All operations admit literal integer entries, at most 64 rows and columns,
and at most 4096 bits per original coefficient. They refuse unsupported
domains, modified built-in owners, source mutation and exhausted work without
falling back to an unchecked algorithm. This bounded capability does not
certify the complete SymPy matrix or principal-ideal-domain interface.
"""

from ..core import (
    Basic, Half, Integer, NegativeOne, One, Rational, Zero,
    _active_callback_session, _charge_nested_if_active, _native,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .dense import ImmutableDenseMatrix, Matrix, MatrixBase
from .sparse import (
    ImmutableSparseMatrix, SparseMatrix, _check_rational_source,
    _rational_source_snapshot, _reduction_binding,
)

_DICT = type.__dict__["__dict__"]
_MRO = type.__dict__["__mro__"]
_CLASSES = (Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix)
_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_NAMES = ("_native_storage", "_matrix_shape", "_surface", "_mutation_generation")
_SLOTS = {base: tuple(_DICT.__get__(base)[name] for name in _NAMES)
          for base in (Matrix, SparseMatrix)}
_ATOM_NAMES = ("_value", "_struct_args", "_surface_args")
_ATOM_SLOTS = tuple(_DICT.__get__(Basic)[name] for name in _ATOM_NAMES)
_MISSING = object()
_NATIVE_EXPR = _native.Expr
_NATIVE_SESSION = _native.SupervisedCallbackSession
_SESSION_FLAGS = tuple(
    _DICT.__get__(_NATIVE_SESSION)[name]
    for name in ("cancelled", "unknown_effect_tainted")
)
_NATIVE = {Matrix: _native.Matrix, SparseMatrix: _native.SparseMatrix}
_NATIVE_NAMES = (
    "shape", "validate_integer_normal_form_source",
    "smith_normal_form", "smith_normal_decomposition",
    "hermite_normal_form", "hermite_normal_decomposition",
)
_NATIVE_METHODS = {
    base: tuple(_DICT.__get__(native)[name] for name in _NATIVE_NAMES)
    for base, native in _NATIVE.items()
}
_OWNERS = tuple((owner, _owner_record(owner), _MRO.__get__(owner))
                for owner in dict.fromkeys(
                    owner for cls in (*_CLASSES, MatrixBase, Basic, *_ATOMS)
                    for owner in (*_MRO.__get__(cls), *_MRO.__get__(type(cls)))
                ))
_NATIVE_VIEWS = tuple(
    (cls, _reduction_binding(cls, "_native"),
     _implementation(_reduction_binding(cls, "_native")))
    for cls in _CLASSES
)


def _interfaces():
    for owner, recorded, parents in _OWNERS:
        current = _MRO.__get__(owner)
        if (len(current) != len(parents)
                or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(owner, recorded)):
            raise NotImplementedError("integer Smith form requires original built-in implementations")
    for cls, original, recorded in _NATIVE_VIEWS:
        current = _reduction_binding(cls, "_native")
        tokens = _implementation(current)
        if (current is not original or len(tokens) != len(recorded)
                or any(a is not b for a, b in zip(tokens, recorded))):
            raise NotImplementedError("integer Smith form requires the original native storage view")
    for cls in _CLASSES:
        base = Matrix if cls in (Matrix, ImmutableDenseMatrix) else SparseMatrix
        if any(_reduction_binding(cls, name) is not slot
               for name, slot in zip(_NAMES, _SLOTS[base])):
            raise NotImplementedError("integer Smith form requires original matrix storage descriptors")
    for base in _SLOTS:
        namespace = _DICT.__get__(_NATIVE[base])
        if any(namespace.get(name) is not method
               for name, method in zip(_NATIVE_NAMES, _NATIVE_METHODS[base])):
            raise NotImplementedError("integer Smith form requires original native interfaces")
    if any(_reduction_binding(Basic, name) is not slot
           for name, slot in zip(_ATOM_NAMES, _ATOM_SLOTS)):
        raise NotImplementedError("integer Smith form requires original scalar storage descriptors")


def _atom_state(value):
    result = []
    for slot in _ATOM_SLOTS:
        try:
            result.append(slot.__get__(value, Basic))
        except AttributeError:
            result.append(_MISSING)
    return tuple(result)


class _NonliteralIntegerSource(NotImplementedError):
    """A source cell is outside the literal-integer lane."""


def _request_unchanged(session):
    if _active_callback_session() is not session:
        raise RuntimeError("integer normal-form request changed during evaluation")
    if session is not None:
        if type(session) is not _NATIVE_SESSION:
            raise TypeError("integer normal forms require an exact native request")
        # These captured Rust getters only read the session state. Ownership
        # checks can execute Python, so no such callback may follow the final
        # original-source check or reopen its publication window.
        if any(flag.__get__(session, _NATIVE_SESSION) for flag in _SESSION_FLAGS):
            raise RuntimeError("cancelled or tainted callback session cannot publish a result")


class _IntegerSource:
    def __init__(self, matrix):
        self.owner = type(matrix)
        if not any(self.owner is cls for cls in _CLASSES):
            raise NotImplementedError("integer Smith form requires a built-in Matrix")
        self.session = _active_callback_session()
        # Terminal active requests stop before any retained/native cell copies.
        _charge_nested_if_active()
        _request_unchanged(self.session)
        if type(matrix) is not self.owner:
            raise RuntimeError("integer matrix owner mutated during source admission")
        _interfaces()
        self.matrix = matrix
        self.base = Matrix if self.owner in (Matrix, ImmutableDenseMatrix) else SparseMatrix
        native, shape, surface, generation = (
            slot.__get__(matrix, self.base) for slot in _SLOTS[self.base]
        )
        if (type(shape) is not tuple or len(shape) != 2
                or any(type(size) is not int or size < 0 for size in shape)
                or type(generation) is not int or generation < 0):
            raise ValueError("integer Smith form has an invalid source snapshot")
        rows, columns = shape
        if max(rows, columns) > 64:
            raise ValueError("integer Smith form supports at most 64 rows and columns")
        native_type = _NATIVE[self.base]
        native_shape, preflight = _NATIVE_METHODS[self.base][:2]
        if type(native) is not native_type:
            raise NotImplementedError("integer Smith form requires literal integer native storage")
        if native_shape.__get__(native, native_type) != shape:
            raise ValueError("integer Smith form native and retained shapes differ")
        self.atoms = []
        self.atom_owners = []
        if surface is not None:
            storage_type = tuple if self.base is Matrix else dict
            if (type(surface) is not storage_type or len(surface) > rows * columns
                    or self.base is Matrix and len(surface) != rows * columns):
                raise ValueError("integer Smith form has invalid retained matrix cells")
            if self.base is SparseMatrix:
                for key in dict.keys(surface):
                    if (type(key) is not tuple or len(key) != 2
                            or type(key[0]) is not int or type(key[1]) is not int
                            or not 0 <= key[0] < rows or not 0 <= key[1] < columns):
                        raise ValueError("integer Smith form has an invalid retained coordinate")
            cells = surface if self.base is Matrix else dict.values(surface)
            for cell in cells:
                owner = type(cell)
                if not any(owner is atom for atom in _ATOMS):
                    raise _NonliteralIntegerSource("integer Smith form requires literal integer source atoms")
                state = _atom_state(cell)
                if type(state[0]) is not _NATIVE_EXPR:
                    raise ValueError("integer Smith form has invalid native scalar storage")
                if any(value is not _MISSING and value is not None
                       and (type(value) is not tuple or len(value) != 0)
                       for value in state[1:]):
                    raise ValueError("integer Smith form requires literal retained atoms")
                self.atoms.append((cell, state))
                self.atom_owners.append(owner)
        # Borrowed native admission bounds every integer before the existing
        # source binder copies any native coefficient or sparse dictionary.
        _charge_nested_if_active(rows * columns + 1)
        _request_unchanged(self.session)
        if type(matrix) is not self.owner:
            raise RuntimeError("integer matrix owner mutated during source admission")
        _interfaces()
        current = tuple(slot.__get__(matrix, self.base) for slot in _SLOTS[self.base])
        if (current[0] is not native or current[1] is not shape
                or current[2] is not surface or type(current[3]) is not int
                or current[3] != generation):
            raise RuntimeError("integer matrix mutated during source admission")
        if surface is not None:
            cells = surface if self.base is Matrix else dict.values(surface)
            if (len(surface) != len(self.atoms)
                    or any(current is not old
                           for current, (old, _) in zip(cells, self.atoms))):
                raise RuntimeError("integer retained cells mutated during source admission")
        for (cell, state), owner in zip(self.atoms, self.atom_owners):
            if (type(cell) is not owner
                    or any(current is not old
                           for current, old in zip(_atom_state(cell), state))):
                raise RuntimeError("integer source atom mutated during source admission")
        preflight(native)
        captured = _rational_source_snapshot(
            matrix, dense=self.base is Matrix, context="integer Smith form",
        )
        if captured is None:
            raise NotImplementedError("integer Smith form requires an exact integer source")
        self.native, self.snapshot = captured
        self.shape = shape
        self.unchanged()

    def unchanged(self):
        _request_unchanged(self.session)
        if type(self.matrix) is not self.owner:
            raise RuntimeError("integer matrix owner mutated during evaluation")
        _interfaces()
        for (cell, _), owner in zip(self.atoms, self.atom_owners):
            if type(cell) is not owner:
                raise RuntimeError("integer source atom owner mutated during evaluation")
        _check_rational_source(self.snapshot)
        for cell, stored in self.atoms:
            if any(current is not previous
                   for current, previous in zip(_atom_state(cell), stored)):
                raise RuntimeError("integer Smith source atom mutated during decomposition")
        _request_unchanged(self.session)

    def calculate(self, decomposition, *, hermite=False):
        self.unchanged()
        offset = 4 if hermite else 2
        form, decompose = _NATIVE_METHODS[self.base][offset:offset + 2]
        rows, columns = self.shape
        if decomposition:
            native = decompose(self.native, session=self.session)
            if type(native) is not tuple or len(native) != 3:
                raise RuntimeError("integer Smith decomposition returned invalid result storage")
            native_type = _NATIVE[self.base]
            shapes = ((rows, columns), (rows, rows), (columns, columns))
            matrix_type, base = self.owner, self.base
        else:
            native = (form(self.native, session=self.session),)
            native_type = _NATIVE[Matrix]
            shapes = (self.shape,)
            matrix_type, base = Matrix, Matrix
        self.unchanged()
        descriptor = _NATIVE_METHODS[base][0]
        if hermite:
            if type(native[0]) is not native_type:
                raise RuntimeError("integer Hermite form returned invalid result storage")
            shape = descriptor.__get__(native[0], native_type)
            if (type(shape) is not tuple or len(shape) != 2
                    or shape[0] != rows or type(shape[1]) is not int
                    or not 0 <= shape[1] <= min(rows, columns)):
                raise RuntimeError("integer Hermite form returned an invalid rank or shape")
            shapes = ((rows, shape[1]), (columns, columns), (columns, columns)) if decomposition else (shape,)
        if any(type(value) is not native_type
               or descriptor.__get__(value, native_type) != shape
               for value, shape in zip(native, shapes)):
            raise RuntimeError("integer Smith decomposition returned invalid output shapes")
        _charge_nested_if_active(sum(row * column for row, column in shapes) + len(shapes))
        result = []
        for value, shape in zip(native, shapes):
            # The already checked storage is installed through captured slots;
            # neither public constructors nor replaceable _native setters run.
            matrix = object.__new__(matrix_type)
            for slot, item in zip(_SLOTS[base], (value, shape, None, 0)):
                slot.__set__(matrix, item)
            result.append(matrix)
        _charge_nested_if_active()
        self.unchanged()
        return tuple(result) if decomposition else result[0]


def smith_normal_form(matrix, domain=None):
    """Return the canonical integer Smith form as a mutable dense Matrix.

    This bounded slice infers ZZ from literal integer entries. Explicit domain
    objects, rational fields and polynomial coefficient rings are unsupported.
    Zero-area rectangular shapes are retained. Every accepted result has been
    checked with integral unimodular transformations against the original input.
    """
    if domain is not None:
        raise NotImplementedError("smith_normal_form supports the inferred integer domain only")
    return _IntegerSource(matrix).calculate(False)


def smith_normal_decomposition(matrix):
    """Return independently checked ``(D, U, V)`` satisfying ``D == U*matrix*V``.

    This FrankenSymPy extension preserves all four built-in matrix classes.
    D has the original rectangular shape; U and V are square integer matrices
    of row and column dimension respectively, with independently checked integer
    inverses. Their concrete entries are not an upstream compatibility promise.
    """
    return _IntegerSource(matrix).calculate(True)


def hermite_normal_form(A, *, D=None, check_rank=False):
    """Return the checked column Hermite form of an integer matrix.

    The output has every source row and exactly one column per independent
    integer-lattice direction. It is a mutable dense Matrix for all four
    built-in input classes, including zero-area or deficient-rank matrices.
    Positive pivots and reduced residues are checked independently of the
    Euclidean generator, together with integral unimodular transformations.

    This bounded slice uses the exact ordinary algorithm. The optional modular
    determinant hint D is not supported; check_rank is accepted as a boolean
    because every result already has its complete rank independently checked.
    """
    if D is not None:
        raise NotImplementedError("hermite_normal_form does not support a modular determinant hint")
    if type(check_rank) is not bool:
        raise TypeError("hermite_normal_form check_rank must be a boolean")
    return _IntegerSource(A).calculate(False, hermite=True)


def hermite_normal_decomposition(matrix):
    """Return checked ``(H, V, V_inverse)`` with ``matrix*V == [0 | H]``.

    H is the canonical column basis, and both integer products V*V_inverse
    and V_inverse*V equal the identity. All three matrices preserve the input
    built-in class. Their shapes are m-by-r, n-by-n and n-by-n, respectively.
    The inverse certifies equality of integer lattices, not just rational spans.
    """
    return _IntegerSource(matrix).calculate(True, hermite=True)


__all__ = [
    "smith_normal_form", "smith_normal_decomposition",
    "hermite_normal_form", "hermite_normal_decomposition",
]
