"""Exact rational/complex inertia and conservative symbolic definiteness.

Rational forms use independent native block-LDL verification. Defined Q(i)
forms through dimension 32 use an independently checked realification of the
original matrix, including its skew-Hermitian part. Other admitted scalar
expressions retain their tri-valued symbolic rules. No principal-minor power
set or floating-point sign guess participates in this implementation.
"""

from ..core import (
    Add, Basic, Dummy, Expr, Half, I, Integer, Mul, NegativeOne, One, Pow,
    Rational, Symbol, Zero, _active_callback_session, _charge_nested_if_active, _native,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .dense import ImmutableDenseMatrix, Matrix, MatrixBase, _least_squares_stored_children
from .sparse import (
    ImmutableSparseMatrix, SparseMatrix, _check_rational_source,
    _rational_source_snapshot, _reduction_binding,
)

_DICT = type.__dict__["__dict__"]
_MRO = type.__dict__["__mro__"]
_CLASSES = (Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix)
_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_SYMBOLIC = (*_ATOMS, Symbol, Dummy, Add, Mul, Pow)
_SCALAR_SLOTS = tuple(_DICT.__get__(Basic)[name]
                      for name in ("_value", "_surface_args", "_struct_args"))
_ARGUMENT_SLOTS = {cls: _DICT.__get__(cls)["_args"] for cls in (Add, Mul, Pow)}
_DECLARATIONS = _DICT.__get__(Symbol)["_assumptions"]
_IMAGINARY_TYPE = type(I)
_IMAGINARY_VALUE = _SCALAR_SLOTS[0].__get__(I, Basic)
_NATIVE_EXPR = _native.Expr
_NATIVE_EQUAL = _reduction_binding(_NATIVE_EXPR, "__eq__")
_NAMES = ("_native_storage", "_matrix_shape", "_surface", "_mutation_generation")
_SLOTS = {base: tuple(_DICT.__get__(base)[name] for name in _NAMES)
          for base in (Matrix, SparseMatrix)}
_OWNERS = tuple((owner, _owner_record(owner), _MRO.__get__(owner))
                for owner in dict.fromkeys(
                    owner for cls in (*_CLASSES, MatrixBase, Basic, Expr, *_SYMBOLIC, _IMAGINARY_TYPE)
                    for owner in (*_MRO.__get__(cls), *_MRO.__get__(type(cls)))
                ))
_READS = ("conjugate", "__add__", "__radd__", "__sub__", "__rsub__",
          "__mul__", "__rmul__", "__neg__", "__truediv__", "__pow__",
          "subs", "atoms", "free_symbols", "name", "p", "q", "_native")
_METHODS = tuple((owner, name, _implementation(_DICT.__get__(owner).get(name)))
                 for owner, *_ in _OWNERS for name in _READS)
_NATIVE = {base: native for base, native in
           ((Matrix, _native.Matrix), (SparseMatrix, _native.SparseMatrix))}
_NATIVE_METHODS = {
    base: tuple(_DICT.__get__(native)[name] for name in
                ("shape", "is_exact_rational", "validate_rational_inertia_source", "quadratic_form_inertia"))
    for base, native in _NATIVE.items()
}
_HERMITIAN_NAMES = ("validate_hermitian_inertia_source", "hermitian_form_inertia")
_HERMITIAN_METHODS = {
    base: tuple(_DICT.__get__(native)[name] for name in _HERMITIAN_NAMES)
    for base, native in _NATIVE.items()
}
_NATIVE_CELLS = {
    Matrix: ("flat", _DICT.__get__(_NATIVE[Matrix])["flat"]),
    SparseMatrix: ("todok", _DICT.__get__(_NATIVE[SparseMatrix])["todok"]),
}
_SCALAR_NATIVE_NAMES = ("raw_args", "is_integer", "is_rational", "is_add", "is_mul", "is_pow",
                        "multiquadratic_source_size")
_SCALAR_NATIVE = tuple(_DICT.__get__(_NATIVE_EXPR)[name] for name in _SCALAR_NATIVE_NAMES)
_POLYNOMIAL_NATIVE_NAMES = ("is_symbol", "polynomial_source_size")
_POLYNOMIAL_NATIVE = tuple(_DICT.__get__(_NATIVE_EXPR)[name]
                           for name in _POLYNOMIAL_NATIVE_NAMES)
_NATIVE_ZERO = _SCALAR_SLOTS[0].__get__(Zero(), Basic)
_FIXED_NATIVE = {cls: _SCALAR_SLOTS[0].__get__(cls(), Basic)
                 for cls in (Zero, One, NegativeOne, Half)}
_UNKNOWN = (None,) * 5


def _interfaces():
    if _reduction_binding(_NATIVE_EXPR, "__eq__") is not _NATIVE_EQUAL:
        raise NotImplementedError("matrix definiteness requires original native scalar equality")
    if any(_DICT.__get__(_NATIVE_EXPR).get(name) is not descriptor
           for name, descriptor in zip(_SCALAR_NATIVE_NAMES, _SCALAR_NATIVE)):
        raise NotImplementedError("matrix definiteness requires original native scalar interfaces")
    if any(_DICT.__get__(_NATIVE_EXPR).get(name) is not descriptor
           for name, descriptor in zip(_POLYNOMIAL_NATIVE_NAMES, _POLYNOMIAL_NATIVE)):
        raise NotImplementedError("matrix source binding requires original native scalar interfaces")
    if any(_reduction_binding(cls, "_assumptions") is not _DECLARATIONS
           for cls in (Symbol, Dummy)):
        raise NotImplementedError("matrix definiteness requires original symbol declarations")
    for owner, recorded, parents in _OWNERS:
        current = _MRO.__get__(owner)
        if (len(current) != len(parents)
                or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(owner, recorded)):
            raise NotImplementedError("matrix definiteness requires original built-in implementations")
    for owner, name, recorded in _METHODS:
        current = _implementation(_DICT.__get__(owner).get(name))
        if (len(current) != len(recorded)
                or any(a is not b for a, b in zip(current, recorded))):
            raise NotImplementedError("matrix definiteness requires original scalar arithmetic")
    for base, slots in _SLOTS.items():
        if any(_reduction_binding(base, name) is not slot
               for name, slot in zip(_NAMES, slots)):
            raise NotImplementedError("matrix definiteness requires original storage descriptors")
        namespace = _DICT.__get__(_NATIVE[base])
        for name, method in zip(
                ("shape", "is_exact_rational", "validate_rational_inertia_source", "quadratic_form_inertia"),
                _NATIVE_METHODS[base]):
            if namespace.get(name) is not method:
                raise NotImplementedError("matrix definiteness requires original native interfaces")
        for name, method in (*zip(_HERMITIAN_NAMES, _HERMITIAN_METHODS[base]),
                             _NATIVE_CELLS[base]):
            if namespace.get(name) is not method:
                raise NotImplementedError("matrix definiteness requires original native interfaces")


def _slot(slot, value):
    try:
        return slot.__get__(value, type(value))
    except AttributeError:
        return None


def _unchanged(matrix, base, state, retained, scalars=()):
    current = tuple(slot.__get__(matrix, base) for slot in _SLOTS[base])
    if (any(current[i] is not state[i] for i in range(3))
            or type(current[3]) is not int or current[3] != state[3]):
        raise RuntimeError("Matrix mutated during definiteness evaluation")
    surface = state[2]
    if type(surface) is dict and (len(surface) != len(retained)
            or any(dict.get(surface, key) is not value for key, value in retained.items())):
        raise RuntimeError("Sparse matrix cells mutated during definiteness evaluation")
    _interfaces()
    for value, slots, stored, declared, facts in scalars:
        if any(_slot(slot, value) is not previous for slot, previous in zip(slots, stored)):
            raise RuntimeError("Matrix scalar storage mutated during definiteness evaluation")
        if declared is not None:
            current = _DECLARATIONS.__get__(value, Symbol)
            if (current is not declared or len(current) != len(facts)
                    or any(dict.get(current, name) is not fact for name, fact in facts)):
                raise RuntimeError("Matrix symbol assumptions mutated during definiteness evaluation")


def _closed(entries, *, gaussian=False, polynomial=False):
    from ..assumptions._query import typed_symbol
    from ..core._opaque_arithmetic import _children

    pending = [(entry, 0) for entry in entries]
    count = expanded = 0
    records = []
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > 4096 or depth > 64:
            raise ValueError("symbolic definiteness source exceeds its expression envelope")
        builtin = any(type(value) is cls for cls in _SYMBOLIC)
        if not builtin and type(value) is not _IMAGINARY_TYPE:
            return None
        argument_slot = _ARGUMENT_SLOTS.get(type(value))
        slots = _SCALAR_SLOTS + (() if argument_slot is None else (argument_slot,))
        stored = tuple(_slot(slot, value) for slot in slots)
        if stored[0] is not None and type(stored[0]) is not _NATIVE_EXPR:
            raise NotImplementedError("matrix definiteness requires original native scalar storage")
        if gaussian or polynomial:
            if type(stored[0]) is not _NATIVE_EXPR:
                raise NotImplementedError("matrix definiteness requires original native scalar storage")
            # Retained handles are mutable independently of matrix storage.
            # Bound each current graph before equality, child copies or any
            # scalar property can traverse a replaced handle.
            size = (_POLYNOMIAL_NATIVE[1] if polynomial else _SCALAR_NATIVE[-1])(stored[0])
            expanded += size
            if expanded > 4096 * 65:
                raise ValueError("Gaussian definiteness binding exceeds its expression envelope")
            _charge_nested_if_active(size + 1)
        # Native child recovery may lift Constant-I into a fresh exact Expr.
        # Admit its native constant identity, never a printed name or an
        # arbitrary generic Expr that happens to render as an imaginary unit.
        if not builtin and (type(stored[0]) is not _NATIVE_EXPR
                            or _NATIVE_EQUAL(stored[0], _IMAGINARY_VALUE) is not True):
            return None
        declared, facts = None, ()
        if type(value) in (Symbol, Dummy):
            declared = _DECLARATIONS.__get__(value, Symbol)
            if (type(declared) is not dict or len(declared) > 64
                    or any(type(name) is not str or type(fact) is not bool
                           or polynomial and len(name) > 256
                           for name, fact in dict.items(declared))):
                raise ValueError("matrix definiteness requires bounded canonical symbol declarations")
            typed_symbol(value)
            facts = tuple(dict.items(declared))
        records.append((value, slots, stored, declared, facts))
        # A scalar can carry more than one retained argument slot. Inspect
        # every present tuple, including one hidden behind another slot's
        # lookup precedence, before granting a native mathematical result.
        retained = []
        for candidate in stored[1:]:
            if candidate is not None:
                if type(candidate) is not tuple:
                    raise ValueError("matrix definiteness requires exact retained argument tuples")
                if not any(candidate is previous for previous in retained):
                    retained.append(candidate)
        if retained:
            if sum(map(len, retained)) > 4096 - count - len(pending):
                raise ValueError("symbolic definiteness source exceeds its expression envelope")
            children = tuple(child for candidate in retained for child in candidate)
        else:
            children = _least_squares_stored_children(value)
            if children is None:
                children = _children(value)
        if children is not None:
            if type(children) is not tuple or count + len(pending) + len(children) > 4096:
                raise ValueError("symbolic definiteness source exceeds its expression envelope")
            pending.extend((child, depth + 1) for child in children)
    # Admit the entire retained graph before an assumption property can inspect
    # any child. Owner checks alone do not bind mutable scalar payloads.
    if (not gaussian or polynomial) and any(
            value.is_commutative is not True or value.is_finite is False
            for value, *_ in records):
        return None
    return tuple(records)


def _all(values):
    values = tuple(values)
    if any(value is False for value in values):
        return False
    if all(value is True for value in values):
        return True
    return None


def _bind_gaussian(native, base, retained, scalars, *, polynomial=False):
    """Bind every retained cell and child before native exact publication.

    Admission has already bounded the borrowed native graph. Checking only the
    root `_value` would miss a changed nested child or held argument tuple.
    Compare retained children directly with raw native children; no evaluation,
    reconstruction, printing or custom scalar callback is needed.
    """
    _charge_nested_if_active(len(retained) + len(scalars) + 1)
    # Admission bounded these immutable handles. Live scalar slots may change
    # meanwhile; publication checks detect that separately, before generation.
    values = {id(scalar): stored[0] for scalar, _slots, stored, _declared, _facts in scalars}
    entries = _NATIVE_CELLS[base][1](native)
    cells = enumerate(retained) if base is Matrix else dict.items(retained)
    for coordinate, entry in cells:
        value = values.get(id(entry))
        expected = entries[coordinate] if base is Matrix else entries.get(coordinate, _NATIVE_ZERO)
        if type(value) is not _NATIVE_EXPR or _NATIVE_EQUAL(value, expected) is not True:
            raise RuntimeError("Matrix retained cells differ from the native Hermitian source")
    if base is SparseMatrix and any(coordinate not in retained for coordinate in entries):
        raise RuntimeError("Matrix retained cells differ from the native Hermitian source")
    raw_args, integer, rational, add, mul, power, _size = _SCALAR_NATIVE
    kinds = {Add: add, Mul: mul, Pow: power}
    for scalar, _slots, stored, _declared, _facts in scalars:
        value = stored[0]
        if type(value) is not _NATIVE_EXPR:
            raise RuntimeError("Matrix retained scalar has no native Hermitian source")
        if any(type(scalar) is atom for atom in _ATOMS):
            integer_value = integer.__get__(value, _NATIVE_EXPR)
            rational_value = rational.__get__(value, _NATIVE_EXPR)
            integer_owner = type(scalar) in (Integer, Zero, One, NegativeOne)
            if not (integer_value if integer_owner else integer_value or rational_value):
                raise RuntimeError("Matrix retained atom differs from its native Hermitian source")
            expected = _FIXED_NATIVE.get(type(scalar))
            if expected is not None and _NATIVE_EQUAL(value, expected) is not True:
                raise RuntimeError("Matrix retained constant differs from its native Hermitian source")
        elif type(scalar) is _IMAGINARY_TYPE:
            if _NATIVE_EQUAL(value, _IMAGINARY_VALUE) is not True:
                raise RuntimeError("Matrix retained imaginary unit differs from its native source")
        elif polynomial and type(scalar) in (Symbol, Dummy):
            if _POLYNOMIAL_NATIVE[0].__get__(value, _NATIVE_EXPR) is not True:
                raise RuntimeError("Matrix retained symbol differs from its native polynomial source")
        else:
            kind = kinds.get(type(scalar))
            if kind is None or kind.__get__(value, _NATIVE_EXPR) is not True:
                raise RuntimeError("Matrix retained scalar kind differs from its native Hermitian source")
        for children in stored[1:]:
            if children is None:
                continue
            expected = raw_args(value)
            if len(children) != len(expected):
                raise RuntimeError("Matrix retained arguments differ from the native Hermitian source")
            for child, original in zip(children, expected):
                child_value = values.get(id(child))
                if (type(child_value) is not _NATIVE_EXPR
                        or _NATIVE_EQUAL(child_value, original) is not True):
                    raise RuntimeError("Matrix retained arguments differ from the native Hermitian source")


def _gaussian_result(matrix, base, state, retained, scalars, session):
    _unchanged(matrix, base, state, retained, scalars)
    result = _HERMITIAN_METHODS[base][1](state[0], session=session)
    _unchanged(matrix, base, state, retained, scalars)
    properties = _inertia_properties(result, state[1][0])
    _charge_nested_if_active()
    return properties


def _inertia_properties(counts, dimension):
    if (type(counts) is not tuple or len(counts) != 3
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != dimension):
        raise RuntimeError("matrix definiteness received incomplete Hermitian inertia")
    positive, negative, _zero = counts
    return (positive == dimension, negative == 0,
            negative == dimension, positive == 0,
            positive > 0 and negative > 0)


def _symbolic(cells, n):
    half = Rational(1, 2)
    h = [[(cells[row*n+column] + cells[column*n+row].conjugate()) * half
          for column in range(n)] for row in range(n)]
    diagonal = [h[i][i] for i in range(n)]
    if all(h[i][j].is_zero is True for i in range(n) for j in range(n) if i != j):
        positive = _all(value.is_positive for value in diagonal)
        nonnegative = _all(value.is_nonnegative for value in diagonal)
        negative = _all(value.is_negative for value in diagonal)
        nonpositive = _all(value.is_nonpositive for value in diagonal)
        indefinite = (True if any(value.is_positive is True for value in diagonal)
                      and any(value.is_negative is True for value in diagonal)
                      else False if nonnegative is True or nonpositive is True else None)
        return positive, nonnegative, negative, nonpositive, indefinite
    if n == 2:
        determinant = h[0][0]*h[1][1] - h[0][1]*h[1][0]
        return (
            _all((diagonal[0].is_positive, determinant.is_positive)),
            _all((*(value.is_nonnegative for value in diagonal), determinant.is_nonnegative)),
            _all((diagonal[0].is_negative, determinant.is_positive)),
            _all((*(value.is_nonpositive for value in diagonal), determinant.is_nonnegative)),
            determinant.is_negative,
        )
    # Necessary diagonal conditions can disprove definiteness; they cannot
    # prove a nondiagonal form positive. Retain unknown instead of guessing.
    positive = False if any(value.is_positive is False for value in diagonal) else None
    nonnegative = False if any(value.is_negative is True for value in diagonal) else None
    negative = False if any(value.is_negative is False for value in diagonal) else None
    nonpositive = False if any(value.is_positive is True for value in diagonal) else None
    indefinite = True if (any(value.is_positive is True for value in diagonal)
                          and any(value.is_negative is True for value in diagonal)) else None
    for i in range(n):
        for j in range(i):
            determinant = h[i][i]*h[j][j] - h[i][j]*h[j][i]
            if determinant.is_negative is True:
                return False, False, False, False, True
    return positive, nonnegative, negative, nonpositive, indefinite


def classify(matrix):
    if not any(type(matrix) is cls for cls in _CLASSES):
        return _UNKNOWN
    session = _active_callback_session()
    _charge_nested_if_active()
    _interfaces()
    base = Matrix if type(matrix) in (Matrix, ImmutableDenseMatrix) else SparseMatrix
    state = tuple(slot.__get__(matrix, base) for slot in _SLOTS[base])
    native, shape, surface, generation = state
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("matrix definiteness has an invalid source snapshot")
    rows, columns = shape
    if rows != columns:
        return (False,) * 5
    native_type = _NATIVE[base]
    native_shape, rational, preflight, calculate = _NATIVE_METHODS[base]
    if native is not None:
        if type(native) is not native_type or native_shape.__get__(native, native_type) != shape:
            raise ValueError("matrix definiteness native and retained shapes differ")
        if rational.__get__(native, native_type):
            # Admit borrowed dimensions and coefficient heights before the
            # existing source binder clones native or retained rational cells.
            preflight(native)
            retained_type = tuple if base is Matrix else dict
            literal = surface is None
            if type(surface) is retained_type and len(surface) <= rows*rows:
                entries = surface if base is Matrix else dict.values(surface)
                literal = all(any(type(entry) is atom for atom in _ATOMS) for entry in entries)
            # Held arithmetic may lower to a rational while retaining a pole.
            # Such input belongs to source-domain admission below; it is not
            # a literal rational source eligible for native publication.
            if literal:
                _charge_nested_if_active(rows*rows + 1)
                captured = _rational_source_snapshot(matrix, dense=base is Matrix, context="matrix definiteness")
                if captured is None:
                    raise RuntimeError("matrix definiteness lost its rational source")
                original, snapshot = captured
                positive, negative, zero = calculate(original, session=session)
                _check_rational_source(snapshot)
                _interfaces()
                if (any(type(count) is not int or count < 0 for count in (positive, negative, zero))
                        or positive + negative + zero != rows):
                    raise RuntimeError("matrix definiteness received incomplete inertia")
                _charge_nested_if_active()
                return positive == rows, negative == 0, negative == rows, positive == 0, positive > 0 and negative > 0
    # This is admission before generation, not a retry after failed algebra.
    # Source parsing checks original nested domains and charges the same active
    # session. Only unsupported grammar selects symbolic fallback. An undefined
    # original source stays unknown; budget, cancellation, coefficient and
    # arithmetic failures remain terminal after generation starts.
    gaussian = False
    if native is not None and rows <= 32:
        try:
            admitted = _HERMITIAN_METHODS[base][0](native, session=session)
        except ZeroDivisionError as error:
            if type(error) is not ZeroDivisionError:
                raise
            _unchanged(matrix, base, state, surface)
            _charge_nested_if_active()
            return _UNKNOWN
        except NotImplementedError as error:
            if type(error) is not NotImplementedError:
                raise
        else:
            if admitted is not None:
                raise RuntimeError("matrix definiteness received invalid Hermitian source admission")
            gaussian = True
        _unchanged(matrix, base, state, surface)
        if gaussian and surface is None:
            return _gaussian_result(matrix, base, state, None, (), session)
    # Symbolic expressions remain shell-owned and retain tri-valued assumptions.
    # Native-only unknown expressions are not reconstructed from printed text.
    if rows > 8 and not gaussian or surface is None:
        return _UNKNOWN
    if base is Matrix:
        if type(surface) is not tuple or len(surface) != rows*rows:
            raise ValueError("matrix definiteness has invalid retained cells")
        retained = surface
        cells = surface
    else:
        if type(surface) is not dict or len(surface) > rows*rows:
            raise ValueError("matrix definiteness has invalid sparse cells")
        for key in dict.keys(surface):
            if (type(key) is not tuple or len(key) != 2
                    or any(type(i) is not int or not 0 <= i < rows for i in key)):
                raise ValueError("matrix definiteness has invalid sparse coordinates")
        retained = dict(surface)
        cells = tuple(retained.get((i, j), Zero()) for i in range(rows) for j in range(rows))
    scalars = _closed(cells, gaussian=gaussian)
    if scalars is None:
        _unchanged(matrix, base, state, retained)
        return _UNKNOWN
    if gaussian:
        _bind_gaussian(native, base, retained, scalars)
        return _gaussian_result(matrix, base, state, retained, scalars, session)
    from ..solvers._checksol import _Work, _check_with_work

    work = _Work()
    domain = _check_with_work(cells, {}, None, work, domain_only=True)
    if work.exhausted:
        raise ValueError("symbolic definiteness source exceeds its domain-check envelope")
    _unchanged(matrix, base, state, retained, scalars)
    if domain is not True:
        return _UNKNOWN
    result = _symbolic(cells, rows)
    _unchanged(matrix, base, state, retained, scalars)
    _charge_nested_if_active()
    return result
