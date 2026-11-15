"""Bind checked rational Jordan decompositions to original public matrices.

Spectrum entries are proposals. The native checker verifies full similarity
and an inverse witness against the original matrix before results are exposed.
This module is imported during package initialization to retain the actual
built-in storage and implementation owners before user code can replace them.
"""

from ..core import Basic, Half, Integer, NegativeOne, One, Rational, Zero, _native
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .dense import ImmutableDenseMatrix, Matrix
from .sparse import (
    _check_rational_source, _rational_source_snapshot, _reduction_binding,
)

_NATIVE = _native.Matrix
_DICT = type.__dict__["__dict__"]
_MRO = type.__dict__["__mro__"]
_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_SLOT_NAMES = ("_native_storage", "_matrix_shape", "_surface", "_mutation_generation")
_SLOTS = tuple(_DICT.__get__(Matrix)[name] for name in _SLOT_NAMES)
_ATOM_NAMES = ("_value", "_struct_args", "_surface_args")
_ATOM_SLOTS = tuple(_DICT.__get__(Basic)[name] for name in _ATOM_NAMES)
_MISSING = object()
_OWNERS = tuple((owner, _owner_record(owner), _MRO.__get__(owner))
                for owner in dict.fromkeys(
                    owner for cls in (Matrix, ImmutableDenseMatrix, Basic, *_ATOMS)
                    for owner in (*_MRO.__get__(cls), *_MRO.__get__(type(cls)))
                ))
_NATIVE_VIEWS = tuple(
    (cls, _reduction_binding(cls, "_native"),
     _implementation(_reduction_binding(cls, "_native")))
    for cls in (Matrix, ImmutableDenseMatrix)
)
_PREFLIGHT = _DICT.__get__(_NATIVE)["validate_rational_jordan_source"]
_DECOMPOSE = _DICT.__get__(_NATIVE)["checked_rational_jordan"]
_NATIVE_SHAPE = _DICT.__get__(_NATIVE)["shape"]
_NATIVE_RATIONAL = _DICT.__get__(_NATIVE)["is_exact_rational"]
_NATIVE_ELIGIBLE = _DICT.__get__(_NATIVE)["is_rational_jordan_input"]


def _interfaces():
    for owner, record, parents in _OWNERS:
        current = _MRO.__get__(owner)
        if (len(current) != len(parents)
                or any(left is not right for left, right in zip(current, parents))
                or not _owner_matches(owner, record)):
            raise NotImplementedError("checked Jordan form requires original built-in implementations")
    for cls, original, recorded in _NATIVE_VIEWS:
        current = _reduction_binding(cls, "_native")
        if current is not original:
            raise NotImplementedError("checked Jordan form requires the original native storage view")
        tokens = _implementation(current)
        if (len(tokens) != len(recorded)
                or any(left is not right for left, right in zip(tokens, recorded))):
            raise NotImplementedError("checked Jordan form requires the original native storage view")
    if (any(_reduction_binding(Matrix, name) is not slot
            for name, slot in zip(_SLOT_NAMES, _SLOTS))
            or any(_reduction_binding(Basic, name) is not slot
                   for name, slot in zip(_ATOM_NAMES, _ATOM_SLOTS))):
        raise NotImplementedError("checked Jordan form requires original storage descriptors")
    namespace = _DICT.__get__(_NATIVE)
    if any(namespace.get(name) is not method for name, method in (
        ("validate_rational_jordan_source", _PREFLIGHT),
        ("checked_rational_jordan", _DECOMPOSE),
        ("shape", _NATIVE_SHAPE), ("is_exact_rational", _NATIVE_RATIONAL),
        ("is_rational_jordan_input", _NATIVE_ELIGIBLE),
    )):
        raise NotImplementedError("checked Jordan form requires original native interfaces")


def _atom_state(value):
    state = []
    for slot in _ATOM_SLOTS:
        try:
            state.append(slot.__get__(value, Basic))
        except AttributeError:
            state.append(_MISSING)
    return tuple(state)


class RationalJordanSource:
    """An original rational source retained across spectral search and lifting."""

    @classmethod
    def select(cls, matrix):
        if type(matrix) not in (Matrix, ImmutableDenseMatrix):
            return None
        _interfaces()
        native, shape, surface, _generation = (
            slot.__get__(matrix, Matrix) for slot in _SLOTS
        )
        if native is None:
            return None
        if type(native) is not _NATIVE:
            raise ValueError("Jordan form has invalid native matrix storage")
        if not _NATIVE_RATIONAL.__get__(native, _NATIVE):
            return None
        if (type(shape) is not tuple or len(shape) != 2
                or any(type(size) is not int or size < 0 for size in shape)
                or shape != _NATIVE_SHAPE.__get__(native, _NATIVE)):
            raise ValueError("Jordan form native and retained shapes differ")
        if surface is not None:
            if type(surface) is not tuple or len(surface) != shape[0] * shape[1]:
                raise ValueError("Jordan form has invalid retained matrix cells")
            if any(type(cell) not in _ATOMS for cell in surface):
                # This is a grammar choice before any spectral calculation;
                # custom/symbolic/held sources retain their generic route.
                return None
        if not _NATIVE_ELIGIBLE.__get__(native, _NATIVE):
            # Dimension/height outside the native envelope is an up-front
            # capability choice, preserving the existing general route.
            return None
        _PREFLIGHT(native)
        result = cls()
        result.matrix = matrix
        result.native, result.snapshot = _rational_source_snapshot(
            matrix, dense=True, context="rational Jordan form",
        )
        result.atoms = []
        for cell in surface or ():
            state = _atom_state(cell)
            if any(value is not _MISSING and value is not None
                   and (type(value) is not tuple or len(value) != 0)
                   for value in state[1:]):
                raise ValueError("Jordan form requires literal rational source atoms")
            result.atoms.append((cell, state))
        result.unchanged()
        return result

    def unchanged(self):
        _interfaces()
        _check_rational_source(self.snapshot)
        for cell, before in self.atoms:
            if any(current is not previous
                   for current, previous in zip(_atom_state(cell), before)):
                raise RuntimeError("Jordan source atom mutated during decomposition")

    def decompose(self, spectrum, calc_transform):
        self.unchanged()
        if type(spectrum) is not dict:
            raise ValueError("Jordan spectrum must be a bounded exact dictionary")
        dimension = self.snapshot[4][0]
        if len(spectrum) > dimension:
            raise ValueError("Jordan spectrum exceeds the matrix dimension")
        if any(type(value) not in _ATOMS for value in spectrum):
            # An irrational spectrum is outside this QQ similarity lane.
            # No native Jordan search has begun and no error was swallowed.
            return None
        hints = []
        for value, multiplicity in spectrum.items():
            if type(multiplicity) is not int or not 1 <= multiplicity <= dimension:
                raise ValueError("Jordan spectrum has an invalid multiplicity")
            state = _atom_state(value)
            if any(item is not _MISSING and item is not None
                   and (type(item) is not tuple or len(item) != 0)
                   for item in state[1:]):
                raise ValueError("Jordan eigenvalues require literal rational atoms")
            hints.append((state[0], multiplicity))
        transform, jordan = _DECOMPOSE(self.native, tuple(hints))
        self.unchanged()
        expected_shape = self.snapshot[4]
        if any(type(value) is not _NATIVE
               or _NATIVE_SHAPE.__get__(value, _NATIVE) != expected_shape
               for value in (transform, jordan)):
            raise RuntimeError("checked Jordan decomposition returned an invalid shape")
        matrix_type = type(self.matrix)
        def lift(native):
            # Constructors dispatch through the replaceable _native setter.
            # Install the already checked immutable storage through the
            # original descriptors, preserving the public matrix class.
            result = object.__new__(matrix_type)
            for slot, value in zip(_SLOTS, (native, expected_shape, None, 0)):
                slot.__set__(result, value)
            return result

        result = (lift(transform), lift(jordan))
        selected = result if calc_transform else result[1]
        self.unchanged()
        return selected
