"""Closed rational matrix-exponential admission without scalar callbacks."""

from types import MemberDescriptorType

from ..core import Half, Integer, NegativeOne, One, Rational, Zero, _native


_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_NAMES = ("_native_storage", "_matrix_shape", "_surface", "_mutation_generation")


def _checked_exponential(matrix, *, sparse):
    from .dense import ImmutableDenseMatrix, Matrix
    from .sparse import ImmutableSparseMatrix, SparseMatrix

    classes = (SparseMatrix, ImmutableSparseMatrix) if sparse else (Matrix, ImmutableDenseMatrix)
    if not any(type(matrix) is cls for cls in classes):
        if sparse:
            raise NotImplementedError("sparse exponential requires a built-in matrix class")
        return None
    owner = SparseMatrix if sparse else Matrix
    namespace = type.__dict__["__dict__"].__get__(owner)
    slots = tuple(namespace.get(name) for name in _NAMES)
    if any(type(slot) is not MemberDescriptorType
           or slot.__objclass__ is not owner or slot.__name__ != name
           for name, slot in zip(_NAMES, slots)):
        raise NotImplementedError("matrix exponential requires original storage descriptors")
    native, shape, surface, generation = (
        slot.__get__(matrix, owner) for slot in slots
    )
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("matrix exponential has an invalid source snapshot")
    rows, columns = shape
    if rows != columns:
        raise ValueError("Matrix exponential only defined for square matrices")
    if rows > 8:
        if sparse:
            raise ValueError("checked sparse matrix exponential supports dimension at most eight")
        return None
    size = rows * columns
    if surface is not None:
        if sparse:
            if type(surface) is not dict or len(surface) > size:
                raise ValueError("matrix exponential has an invalid stored-cell snapshot")
            for key in dict.keys(surface):
                if (type(key) is not tuple or len(key) != 2
                        or any(type(index) is not int for index in key)
                        or not (0 <= key[0] < rows and 0 <= key[1] < columns)):
                    raise ValueError("matrix exponential has an invalid stored coordinate")
            entries = dict.values(surface)
        else:
            if type(surface) is not tuple or len(surface) != size:
                raise ValueError("matrix exponential has an invalid retained-cell snapshot")
            entries = surface
        if any(not any(type(entry) is atom for atom in _ATOMS) for entry in entries):
            if sparse:
                raise NotImplementedError(
                    "checked sparse matrix exponential requires literal exact rational entries"
                )
            return None
    native_class = _native.SparseMatrix if sparse else _native.Matrix
    if native is None:
        if sparse:
            raise NotImplementedError(
                "checked sparse matrix exponential requires closed exact rational entries"
            )
        return None
    if type(native) is not native_class or native.shape != shape:
        raise ValueError("matrix exponential native and retained shapes differ")
    if not native.is_exact_rational:
        if sparse:
            raise NotImplementedError(
                "checked sparse matrix exponential requires exact rational entries"
            )
        return None
    # Once admitted, every native refusal is terminal. A second algorithm
    # must not start a fresh work ledger after transform search has begun.
    result = native.exp()
    current = tuple(slot.__get__(matrix, owner) for slot in slots)
    if (current[0] is not native or current[1] is not shape
            or current[2] is not surface or type(current[3]) is not int
            or current[3] != generation):
        raise RuntimeError("Matrix mutated during matrix exponential evaluation")
    return result
