"""Sparse matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from __future__ import annotations

import sys
from collections import OrderedDict
from types import MemberDescriptorType
from typing import Any, Dict, Tuple
from ..core import (
    Basic, Expr, Half, Integer, NegativeOne, One, Rational, Symbol, Zero,
    _native, _native_expr, _wrap, _SYMBOL_CACHE, _surface_symbols,
)
from .dense import Matrix, MatrixBase, ImmutableDenseMatrix, _matrix_dimensions, _surface_entry

_NativeSparseMatrix = _native.SparseMatrix


# Sparse dimensions do not impose a dense-area bound. Only explicit cells and
# operations which actually enumerate that area carry storage/work limits.
_MAX_SPARSE_ENTRIES = 262_144
_MAX_SPARSE_PRODUCTS = 1_000_000
_ZERO = Integer(0)
_REDUCTION_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_REDUCTION_VALUE_SLOT = Basic.__dict__["_value"]
_REDUCTION_MRO = type.__dict__["__mro__"]
_REDUCTION_DICT = type.__dict__["__dict__"]
_REDUCTION_NATIVE_EXPR = _native.Expr
_REDUCTION_NATIVE_EQUAL = _REDUCTION_NATIVE_EXPR.__eq__
_REDUCTION_NATIVE_ZERO = _REDUCTION_VALUE_SLOT.__get__(_ZERO, Basic)
_AFFINE_NATIVE_SYMBOL = _native.py_symbol
_AFFINE_SYMBOL_ASSUMPTIONS = Symbol.__dict__["_assumptions"]
_REDUCTION_NATIVE_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_NativeSparseMatrix)[name]
    for name in (
        "shape", "is_exact_rational", "todok", "to_dense",
        "rref", "rank", "nullspace", "columnspace", "solve", "gauss_jordan_solve",
    )
}
_SOLVE_NATIVE_DENSE = _native.Matrix
_SOLVE_DENSE_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_SOLVE_NATIVE_DENSE)[name]
    for name in ("shape", "is_exact_rational", "flat", "to_sparse")
}
_REDUCTION_STORAGE_NAMES = (
    "_native_storage", "_matrix_shape", "_surface", "_mutation_generation",
)


def _sparse_dimensions(rows, cols):
    rows, cols = int(rows), int(cols)
    if rows < 0 or cols < 0:
        raise OverflowError("can't convert negative int to unsigned")
    if rows > 2*sys.maxsize + 1 or cols > 2*sys.maxsize + 1:
        raise OverflowError("sparse matrix dimension does not fit the native size type")
    return rows, cols


def _is_exact_zero(value):
    """Recognize the exact zero singleton without consulting expression hooks."""
    return value is _ZERO


def _stored_entries(entries):
    if len(entries) > _MAX_SPARSE_ENTRIES:
        raise ValueError("sparse matrix exceeds the stored-entry limit")
    return entries


def _admit_native_cell(value):
    # Exact atomic kinds cannot fabricate a Python function class. They also
    # remain valid symbolic inputs beyond the numerical compiler's bit limits.
    if value.is_symbol or value.is_integer or value.is_rational:
        return
    from ..core._opaque_arithmetic import _validate_native_region

    _validate_native_region(value)


def _sparse_entry(value):
    if type(value) is _native.Expr:
        _admit_native_cell(value)
    return _surface_entry(value)


def _capture_sources(*matrices):
    return tuple((matrix, matrix._mutation_generation, matrix._surface,
                  matrix._native_storage, matrix._matrix_shape)
                 for matrix in matrices)


def _check_sources(sources):
    for matrix, generation, surface, native, shape in sources:
        if (matrix._mutation_generation != generation
                or matrix._surface is not surface
                or matrix._native_storage is not native
                or matrix._matrix_shape != shape):
            raise RuntimeError("Matrix mutated during callback traversal")


def _checked_entry(value, sources):
    _check_sources(sources)
    result = _sparse_entry(value)
    # A callback's result can itself carry an effectful _sympy_ converter.
    _check_sources(sources)
    return result


def _reduction_options(iszerofunc=None, simplify=False):
    # Exact rational reduction has no undecidable zero tests. Callbacks still
    # have observable Python effects and must never be silently discarded.
    if iszerofunc is not None or type(simplify) is not bool:
        raise NotImplementedError(
            "sparse row reduction with custom zero or simplification callbacks "
            "requires a supervised override lane"
        )


def _reduction_binding(cls, name):
    """Resolve a descriptor without invoking a supplied attribute accessor."""
    for owner in _REDUCTION_MRO.__get__(cls):
        namespace = _REDUCTION_DICT.__get__(owner)
        if name in namespace:
            return namespace[name]
    return None


def _reduction_default_access(cls):
    return (_reduction_binding(cls, "__getattribute__") is object.__getattribute__
            and _reduction_binding(cls, "__getattr__") is None)


def _rational_source_snapshot(matrix, *, dense=False, context="sparse row reduction"):
    """Capture closed literal cells without consulting replaceable accessors."""
    base = Matrix if dense else SparseMatrix
    accepted = (Matrix, ImmutableDenseMatrix) if dense else (SparseMatrix, ImmutableSparseMatrix)
    if not any(type(matrix) is cls for cls in accepted):
        return None
    namespace = _REDUCTION_DICT.__get__(base)
    slots = tuple(namespace.get(name) for name in _REDUCTION_STORAGE_NAMES)
    if any(type(slot) is not MemberDescriptorType
           or slot.__objclass__ is not base or slot.__name__ != name
           for name, slot in zip(_REDUCTION_STORAGE_NAMES, slots)):
        raise NotImplementedError(f"{context} requires original storage descriptors")
    if (not _reduction_default_access(type(matrix))
            or any(_reduction_binding(type(matrix), name) is not slot
                   for name, slot in zip(_REDUCTION_STORAGE_NAMES, slots))):
        raise NotImplementedError(f"{context} requires original matrix accessors")
    native, shape, surface, generation = (
        slot.__get__(matrix, base) for slot in slots
    )
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError(f"{context} has an invalid source snapshot")
    if native is None:
        return None
    native_type = _SOLVE_NATIVE_DENSE if dense else _NativeSparseMatrix
    members = _SOLVE_DENSE_MEMBERS if dense else _REDUCTION_NATIVE_MEMBERS
    native_namespace = _REDUCTION_DICT.__get__(native_type)
    if (not _reduction_default_access(native_type)
            or any(native_namespace.get(name) is not descriptor
                   for name, descriptor in members.items())):
        raise NotImplementedError(f"{context} requires original native interfaces")
    if (type(native) is not native_type
            or members["shape"].__get__(native, native_type) != shape):
        raise ValueError(f"{context} native and retained shapes differ")
    if not members["is_exact_rational"].__get__(native, native_type):
        return None
    atoms = []
    retained = None
    if surface is not None:
        storage_type = tuple if dense else dict
        if type(surface) is not storage_type or len(surface) > _MAX_SPARSE_ENTRIES:
            raise ValueError(f"{context} has an invalid stored-cell snapshot")
        if dense and len(surface) != shape[0]*shape[1]:
            raise ValueError(f"{context} retained cells do not match its shape")
        if Basic.__dict__.get("_value") is not _REDUCTION_VALUE_SLOT:
            raise NotImplementedError(f"{context} requires original scalar storage")
        retained = tuple(surface) if dense else dict(surface)
        entries = members["flat"](native) if dense else members["todok"](native)
        cells = enumerate(retained) if dense else retained.items()
        for coordinate, entry in cells:
            if not dense and (type(coordinate) is not tuple or len(coordinate) != 2
                    or any(type(index) is not int for index in coordinate)
                    or not (0 <= coordinate[0] < shape[0] and 0 <= coordinate[1] < shape[1])):
                raise ValueError(f"{context} has an invalid stored coordinate")
            if not any(type(entry) is atom for atom in _REDUCTION_ATOMS):
                raise NotImplementedError(
                    f"checked {context} requires literal exact rational entries"
                )
            if (not _reduction_default_access(type(entry))
                    or _reduction_binding(type(entry), "_value") is not _REDUCTION_VALUE_SLOT):
                raise NotImplementedError(f"{context} requires original scalar accessors")
            value = _REDUCTION_VALUE_SLOT.__get__(entry, Basic)
            stored = entries[coordinate] if dense else entries.get(coordinate, _REDUCTION_NATIVE_ZERO)
            if (type(value) is not _REDUCTION_NATIVE_EXPR
                    or _REDUCTION_NATIVE_EQUAL(value, stored) is not True):
                raise RuntimeError("Matrix retained cells differ from the native snapshot")
            atoms.append((entry, value))
        if not dense and any(coordinate not in retained for coordinate in entries):
            raise RuntimeError("Matrix retained cells differ from the native snapshot")
    snapshot = (matrix, base, slots, native, shape, surface, generation, retained, atoms, dense)
    return native, snapshot


def _check_rational_source(snapshot):
    matrix, base, slots, native, shape, surface, generation, retained, atoms, dense = snapshot
    current = tuple(slot.__get__(matrix, base) for slot in slots)
    if (current[0] is not native or current[1] is not shape
            or current[2] is not surface or type(current[3]) is not int
            or current[3] != generation):
        raise RuntimeError("Matrix mutated during sparse linear algebra")
    if retained is not None:
        changed = (any(current is not old for current, old in zip(surface, retained))
                   if dense else any(surface.get(coordinate) is not entry
                                     for coordinate, entry in retained.items()))
        if (len(surface) != len(retained) or changed
                or any(_REDUCTION_VALUE_SLOT.__get__(entry, Basic) is not value
                       for entry, value in atoms)):
            raise RuntimeError("Matrix retained cells mutated during sparse linear algebra")


def _checked_sparse_reduction(matrix, operation):
    """Run the rational lane only against an intact, closed source snapshot."""
    captured = _rational_source_snapshot(matrix)
    if captured is None:
        return None
    native, snapshot = captured
    # Once native generation starts, all failures are terminal. In particular
    # resource exhaustion must not trigger a dense retry with a fresh budget.
    result = _REDUCTION_NATIVE_MEMBERS[operation](native)
    _check_rational_source(snapshot)
    return result


def _checked_sparse_lift(native, cls):
    """Publish checked cells without dispatching through mutable constructors."""
    layout = next((item for item in _REDUCTION_RESULT_LAYOUTS if item[0] is cls), None)
    if layout is None:
        raise NotImplementedError("checked sparse results require a built-in matrix class")
    _, base, slots, hooks = layout
    if (type(cls) is not type or not _reduction_default_access(cls)
            or any(_reduction_binding(cls, name) is not slot
                   for name, slot in zip(_REDUCTION_STORAGE_NAMES, slots))
            or any(_reduction_binding(cls, name) is not original
                   for name, original in hooks)):
        raise NotImplementedError("checked sparse results require original result construction interfaces")
    if type(native) is not _NativeSparseMatrix:
        raise TypeError("checked sparse result requires exact native sparse storage")
    shape = _REDUCTION_NATIVE_MEMBERS["shape"].__get__(native, _NativeSparseMatrix)
    payload = native
    if base is Matrix:
        _matrix_dimensions(*shape)
        payload = _REDUCTION_NATIVE_MEMBERS["to_dense"](native)
    result = object.__new__(cls)
    for slot, value in zip(slots, (payload, shape, None, 0)):
        slot.__set__(result, value)
    return result


def _affine_symbol_interfaces():
    if (type(Symbol) is not type or not _reduction_default_access(Symbol)
            or any(_reduction_binding(Symbol, name) is not original
                   for name, original in _AFFINE_SYMBOL_INTERFACES)):
        raise NotImplementedError("affine parameters require original Symbol interfaces")


def _affine_parameter_symbols(parameters):
    """Bind native tau atoms to intact cached symbols without scalar callbacks."""
    rows, cols = _REDUCTION_NATIVE_MEMBERS["shape"].__get__(parameters, _NativeSparseMatrix)
    entries = _REDUCTION_NATIVE_MEMBERS["todok"](parameters)
    count = rows*cols
    if count > _MAX_SPARSE_ENTRIES or len(entries) != count:
        raise RuntimeError("native affine parameter matrix has invalid storage")
    if not count:
        return ()
    _affine_symbol_interfaces()
    if type(_SYMBOL_CACHE) is not dict or type(_surface_symbols) is not OrderedDict:
        raise NotImplementedError("affine parameters require original symbol registries")
    bindings = []
    for index in range(count):
        name = f"tau{index}"
        native = entries.get(divmod(index, cols))
        expected = _AFFINE_NATIVE_SYMBOL(name)
        if (type(native) is not _REDUCTION_NATIVE_EXPR
                or _REDUCTION_NATIVE_EQUAL(native, expected) is not True):
            raise RuntimeError("native affine parameter identity is invalid")
        cached = dict.get(_SYMBOL_CACHE, (name, ()))
        registered = OrderedDict.get(_surface_symbols, (name, None))
        if cached is not None and registered is not None and cached is not registered:
            raise RuntimeError("affine parameter registries disagree")
        symbol = cached if cached is not None else registered
        if symbol is None:
            symbol = object.__new__(Symbol)
            _REDUCTION_VALUE_SLOT.__set__(symbol, native)
            _AFFINE_SYMBOL_ASSUMPTIONS.__set__(symbol, {})
        if type(symbol) is not Symbol:
            raise NotImplementedError("affine parameter cache requires exact Symbol atoms")
        value = _REDUCTION_VALUE_SLOT.__get__(symbol, Basic)
        assumptions = _AFFINE_SYMBOL_ASSUMPTIONS.__get__(symbol, Symbol)
        if (type(value) is not _REDUCTION_NATIVE_EXPR
                or _REDUCTION_NATIVE_EQUAL(value, expected) is not True
                or type(assumptions) is not dict or len(assumptions) != 0):
            raise RuntimeError("cached affine parameter differs from its native identity")
        bindings.append((name, symbol, value, assumptions, cached, registered))
    # Validate the complete cache before publishing any new parameter owner.
    for name, symbol, value, assumptions, cached, registered in bindings:
        if (dict.get(_SYMBOL_CACHE, (name, ())) is not cached
                or OrderedDict.get(_surface_symbols, (name, None)) is not registered):
            raise RuntimeError("affine parameter registry mutated during construction")
    for name, symbol, _, _, _, _ in bindings:
        dict.__setitem__(_SYMBOL_CACHE, (name, ()), symbol)
        OrderedDict.__setitem__(_surface_symbols, (name, None), symbol)
    return tuple(bindings)


def _check_affine_parameters(bindings):
    if not bindings:
        return
    _affine_symbol_interfaces()
    for name, symbol, value, assumptions, _, _ in bindings:
        if (_REDUCTION_VALUE_SLOT.__get__(symbol, Basic) is not value
                or _AFFINE_SYMBOL_ASSUMPTIONS.__get__(symbol, Symbol) is not assumptions
                or len(assumptions) != 0
                or dict.get(_SYMBOL_CACHE, (name, ())) is not symbol
                or OrderedDict.get(_surface_symbols, (name, None)) is not symbol):
            raise RuntimeError("affine parameters mutated during construction")


def _checked_gauss_jordan_solve(matrix, rhs, freevar=False):
    """Shared dense/sparse affine entry point over one checked sparse family."""
    if type(freevar) is not bool:
        raise TypeError("freevar must be a bool value")
    accepted = (Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix)
    if not any(type(matrix) is cls for cls in accepted):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    if not any(type(rhs) is cls for cls in accepted):
        raise TypeError("gauss_jordan_solve requires a built-in Matrix right-hand side")
    dense_left = type(matrix) is Matrix or type(matrix) is ImmutableDenseMatrix
    dense_right = type(rhs) is Matrix or type(rhs) is ImmutableDenseMatrix
    left = _rational_source_snapshot(matrix, dense=dense_left, context="exact affine solve")
    right = _rational_source_snapshot(rhs, dense=dense_right, context="exact affine solve")
    if left is None or right is None:
        raise ValueError("exact affine solve supports exact rational entries only")
    coefficients, source_snapshot = left
    rhs_native, rhs_snapshot = right
    rows, unknowns = source_snapshot[4]
    rhs_rows, rhs_columns = rhs_snapshot[4]
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")
    if rhs_columns == 0 and unknowns:
        # The pinned API refuses this surface, with different internal errors
        # depending on rank. Keep the existing explicit bounded refusal.
        raise ValueError("exact affine solve requires a nonempty right-hand side")
    if dense_left:
        coefficients = _SOLVE_DENSE_MEMBERS["to_sparse"](coefficients)
    if dense_right:
        rhs_native = _SOLVE_DENSE_MEMBERS["to_sparse"](rhs_native)
    _check_rational_source(source_snapshot)
    _check_rational_source(rhs_snapshot)
    native_namespace = _REDUCTION_DICT.__get__(_NativeSparseMatrix)
    if (not _reduction_default_access(_NativeSparseMatrix)
            or any(native_namespace.get(name) is not descriptor
                   for name, descriptor in _REDUCTION_NATIVE_MEMBERS.items())):
        raise NotImplementedError("exact affine solve requires original native interfaces")
    family = _REDUCTION_NATIVE_MEMBERS["gauss_jordan_solve"](coefficients, rhs_native)
    _check_rational_source(source_snapshot)
    _check_rational_source(rhs_snapshot)
    if family is None:
        raise ValueError("Linear system has no solution")
    solution_native, parameters_native, free_columns = family
    shape = _REDUCTION_NATIVE_MEMBERS["shape"]
    if (shape.__get__(solution_native, _NativeSparseMatrix) != (unknowns, rhs_columns)
            or shape.__get__(parameters_native, _NativeSparseMatrix) != (len(free_columns), rhs_columns)):
        raise RuntimeError("native affine family has invalid output shapes")
    bindings = _affine_parameter_symbols(parameters_native)
    solution = _checked_sparse_lift(solution_native, type(matrix))
    parameters = _checked_sparse_lift(parameters_native, type(matrix))
    _check_rational_source(source_snapshot)
    _check_rational_source(rhs_snapshot)
    _check_affine_parameters(bindings)
    return ((solution, parameters, list(free_columns)) if freevar
            else (solution, parameters))


class SparseMatrix(MatrixBase):
    """Shell-owned sparse cells with a native snapshot for closed expressions."""

    __slots__ = ("_native_storage", "_matrix_shape", "_surface", "_mutation_generation")

    def __init__(self, *args: Any):
        self._mutation_generation = 0
        if len(args) == 1:
            arg = args[0]
            if type(arg) is _NativeSparseMatrix:
                self._native = arg
                return
            if isinstance(arg, SparseMatrix):
                self._native_storage = arg._native_storage
                self._matrix_shape = arg._matrix_shape
                self._surface = None if arg._surface is None else dict(arg._surface)
                return
            if isinstance(arg, Matrix):
                if arg._surface is None and arg._native_storage is not None:
                    self._native = arg._native_storage.to_sparse()
                    return
                self._install_surface(arg.rows, arg.cols, {
                    (r, c): arg[r, c] for r in range(arg.rows) for c in range(arg.cols)
                })
                return
            if isinstance(arg, (list, tuple)):
                if not arg:
                    self._install_surface(0, 0, {})
                    return
                if isinstance(arg[0], (list, tuple)):
                    rows, cols = len(arg), len(arg[0])
                    _matrix_dimensions(rows, cols)
                    if any(len(row) != cols for row in arg):
                        raise ValueError("Row length mismatch in sparse matrix constructor")
                    self._install_surface(rows, cols, {
                        (r, c): value for r, row in enumerate(arg) for c, value in enumerate(row)
                    })
                else:
                    _stored_entries(arg)
                    self._install_surface(len(arg), 1, {(r, 0): value for r, value in enumerate(arg)})
                return
            raise TypeError(f"Cannot construct SparseMatrix from {type(arg)}")
        if len(args) == 2:
            rows, cols = _sparse_dimensions(*args)
            self._install_surface(rows, cols, {})
            return
        if len(args) != 3:
            raise TypeError(f"SparseMatrix constructor takes 1, 2, or 3 arguments, got {len(args)}")
        rows, cols = _sparse_dimensions(*args[:2])
        source = args[2]
        if isinstance(source, dict):
            _stored_entries(source)
            self._install_surface(rows, cols, source)
        elif isinstance(source, (list, tuple)):
            if len(source) != rows*cols:
                raise ValueError("sparse matrix shape does not match its flat storage")
            _stored_entries(source)
            self._install_surface(rows, cols, {
                divmod(index, cols): value for index, value in enumerate(source)
            })
        elif callable(source):
            # A callback is invoked for every cell, so this constructor really
            # does have dense work even when its eventual result has few cells.
            _matrix_dimensions(rows, cols)
            self._install_surface(rows, cols, {
                (r, c): source(r, c) for r in range(rows) for c in range(cols)
            })
        else:
            raise TypeError("Third argument to SparseMatrix must be a dict, sequence, or callable")

    @property
    def _native(self):
        if self._native_storage is None:
            raise NotImplementedError(
                "sparse matrix contains opaque Python expressions; this operation requires closed native cells"
            )
        return self._native_storage

    @_native.setter
    def _native(self, value):
        if type(value) is not _NativeSparseMatrix:
            raise TypeError("sparse native storage must be an exact native SparseMatrix")
        self._native_storage = value
        self._matrix_shape = value.shape
        self._surface = None

    def _install_surface(self, rows, cols, entries, *, mutate=False,
                         expected_generation=None, expected_sources=None):
        """Prepare every retained cell and native admission before publication."""
        from ..core._opaque_arithmetic import contains_opaque

        rows, cols = _sparse_dimensions(rows, cols)
        _stored_entries(entries)
        retained = {}
        opaque = False
        for coordinate, value in entries.items():
            if not isinstance(coordinate, tuple) or len(coordinate) != 2:
                raise TypeError("sparse coordinates must be (row, col) pairs")
            row, col = map(int, coordinate)
            if not (0 <= row < rows and 0 <= col < cols):
                raise IndexError("Matrix index out of bounds")
            value = _sparse_entry(value)
            # Never ask a custom expression's equality/printing/assumptions
            # hooks whether it is zero. A held or unknown zero stays retained.
            if _is_exact_zero(value):
                continue
            opaque = contains_opaque(value) or opaque
            retained[(row, col)] = value
        # Upstream sparse storage groups by the first occurrence of each row,
        # then by first-seen columns in that row. This determines applyfunc's
        # callback order without sorting or traversing the implicit zeros.
        grouped = {}
        for (row, col), value in retained.items():
            grouped.setdefault(row, {})[col] = value
        retained = {(row, col): value for row, values in grouped.items()
                    for col, value in values.items()}
        native = None
        if not opaque:
            native = _NativeSparseMatrix(rows, cols, {
                coordinate: _native_expr(value) for coordinate, value in retained.items()
            })
        if expected_generation is not None and self._mutation_generation != expected_generation:
            raise RuntimeError("SparseMatrix mutated during callback traversal")
        if expected_sources is not None:
            _check_sources(expected_sources)
        self._native_storage = native
        self._matrix_shape = (rows, cols)
        self._surface = retained
        if mutate:
            self._mutation_generation += 1

    def _new(self, *args):
        return type(self)(*args)

    def _operation_entries(self):
        """Capture stored cells, admitting native-only regions before lifting."""
        if self._surface is not None:
            return dict(self._surface)
        entries = self._native_storage.todok()
        for value in entries.values():
            _admit_native_cell(value)
        return {coordinate: _wrap(value) for coordinate, value in entries.items()}

    def _finish_operation(self, rows, cols, entries, sources, cls=None):
        _check_sources(sources)
        result = (type(self) if cls is None else cls)(rows, cols, entries)
        _check_sources(sources)
        return result

    def _algebra_class(self, other):
        if isinstance(self, ImmutableSparseMatrix):
            return type(self)
        if isinstance(other, ImmutableSparseMatrix):
            return type(other)
        return type(self)

    @property
    def mutation_generation(self):
        return self._mutation_generation

    @property
    def shape(self) -> Tuple[int, int]:
        return self._matrix_shape

    @property
    def rows(self) -> int:
        return self._matrix_shape[0]

    @property
    def cols(self) -> int:
        return self._matrix_shape[1]

    @property
    def nnz(self) -> int:
        return len(self._surface) if self._surface is not None else self._native_storage.nnz

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    @property
    def is_zero_matrix(self) -> bool:
        return self.nnz == 0 or all(v == 0 for v in self.todok().values())

    @property
    def is_identity(self) -> bool:
        return bool(self.to_dense().is_identity)

    @property
    def is_symmetric(self) -> bool:
        return bool(self.to_dense().is_symmetric)

    @property
    def is_anti_symmetric(self) -> bool:
        return bool(self.to_dense().is_anti_symmetric)

    @property
    def is_skew_symmetric(self) -> bool:
        return self.is_anti_symmetric

    @property
    def is_diagonal(self) -> bool:
        return bool(self.to_dense().is_diagonal)

    def is_nilpotent(self) -> bool:
        return bool(self.to_dense().is_nilpotent())

    @property
    def T(self) -> "SparseMatrix":
        return self.transpose()

    def transpose(self) -> "SparseMatrix":
        if self._surface is None:
            return self._new(self._native_storage.transpose())
        return self._new(self.cols, self.rows, {(c, r): value for (r, c), value in self.todok().items()})

    def trace(self) -> Any:
        if not self.is_square:
            raise ValueError(f"Matrix must be square: shape is {self.rows}x{self.cols}")
        if (self._surface is None
                or self._native_storage is not None and self._native_storage.is_exact_rational):
            return _sparse_entry(self._native_storage.trace())
        from ..core import Add

        sources = _capture_sources(self)
        result = Add(*(value for (row, col), value in self._surface.items()
                       if row == col))
        _check_sources(sources)
        return result

    def to_dense(self) -> Matrix:
        _matrix_dimensions(self.rows, self.cols)
        cls = ImmutableDenseMatrix if isinstance(self, ImmutableSparseMatrix) else Matrix
        if self._surface is None:
            return cls(self._native_storage.to_dense())
        entries = self.todok()
        return cls(self.rows, self.cols, [
            entries.get((r, c), _ZERO) for r in range(self.rows) for c in range(self.cols)
        ])

    def as_mutable(self) -> "SparseMatrix":
        return SparseMatrix(self)

    def as_immutable(self) -> "ImmutableSparseMatrix":
        return ImmutableSparseMatrix(self)

    def exp(self) -> "SparseMatrix":
        """Return a checked exact rational exponential through dimension eight.

        The result preserves this matrix class. Rational linear/quadratic
        spectral factors supply a complete normalized fundamental matrix;
        bounded result fill needs no dense source conversion. Unsupported
        spectra and exhausted work refuse explicitly.
        """
        from ._exponential import _checked_exponential

        return self._new(_checked_exponential(self, sparse=True))

    def det(self, method: Any = None) -> Any:
        if self._native.is_exact_rational:
            return _wrap(self._native.det())
        return self.to_dense().det(method=method)

    def inv(self, method: Any = None, **kwargs: Any) -> "SparseMatrix":
        if self._native.is_exact_rational:
            return self.solve(SparseMatrix.eye(self.rows))
        return self._new(self.to_dense().inv(method=method, **kwargs))

    def solve(self, b: Any) -> "SparseMatrix":
        """Solve a square exact rational sparse system without dense conversion."""
        # SparseMatrix.solve and LUsolve have different rectangular contracts
        # in the pinned profile. The native solve also serves LUsolve below.
        if self.rows < self.cols:
            raise ValueError("Under-determined system.")
        if self.rows > self.cols:
            raise ValueError(
                "For over-determined system, M, having more rows than columns, "
                "try M.solve_least_squares(rhs)."
            )
        if isinstance(b, SparseMatrix):
            sparse_rhs = b
        elif isinstance(b, Matrix):
            sparse_rhs = b.to_sparse()
        elif not hasattr(b, "to_dense"):
            sparse_rhs = SparseMatrix(Matrix(b))
        else:
            sparse_rhs = SparseMatrix(b.to_dense())
        if self._native.is_exact_rational and sparse_rhs._native.is_exact_rational:
            return self._new(self._native.solve(sparse_rhs._native))
        return self._new(self.to_dense().solve(sparse_rhs.to_dense()))

    def LUsolve(self, b: Any):
        """Solve a square or consistent tall rational system, keeping RHS class.

        Closed rational inputs use checked sparse elimination without a dense
        intermediate. Underdetermined, inconsistent, and rank-deficient
        systems are refused; no arbitrary free parameters are introduced.
        """
        if not any(type(self) is cls for cls in (SparseMatrix, ImmutableSparseMatrix)):
            raise NotImplementedError("custom matrix subclasses require a supervised override lane")
        rhs_sparse = any(type(b) is cls for cls in (SparseMatrix, ImmutableSparseMatrix))
        if not rhs_sparse and not any(type(b) is cls for cls in (Matrix, ImmutableDenseMatrix)):
            raise TypeError("LUsolve requires a built-in Matrix right-hand side")
        left = _rational_source_snapshot(self, context="sparse LUsolve")
        right = _rational_source_snapshot(b, dense=not rhs_sparse, context="sparse LUsolve")
        if left is None or right is None:
            # Preserve the existing bounded square symbolic lane. Rectangular
            # symbolic pivots require a separate supervised solve contract.
            if self.rows != self.cols:
                raise NotImplementedError("rectangular sparse LUsolve requires exact rational entries")
            result = self.solve(b)
            return type(b)(result) if rhs_sparse else type(b)(result.to_dense())
        coefficients, source_snapshot = left
        rhs, rhs_snapshot = right
        rows, unknowns = source_snapshot[4]
        rhs_rows, _ = rhs_snapshot[4]
        if rows != rhs_rows:
            raise ValueError("The matrices have incompatible number of rows")
        if rows < unknowns:
            raise NotImplementedError("Underdetermined systems not supported.")
        if not rhs_sparse:
            rhs = _SOLVE_DENSE_MEMBERS["to_sparse"](rhs)
        _check_rational_source(source_snapshot)
        _check_rational_source(rhs_snapshot)
        solution = _REDUCTION_NATIVE_MEMBERS["solve"](coefficients, rhs)
        _check_rational_source(source_snapshot)
        _check_rational_source(rhs_snapshot)
        result = _checked_sparse_lift(solution, type(b))
        _check_rational_source(source_snapshot)
        _check_rational_source(rhs_snapshot)
        return result

    def gauss_jordan_solve(self, B, freevar=False):
        """Return the complete checked exact-rational family, retaining sparsity.

        All right-hand sides share one elimination and one complete nullspace.
        The solution and parameter matrices retain this coefficient class;
        each RHS receives its own independent row-major tau parameters.
        """
        return _checked_gauss_jordan_solve(self, B, freevar)

    def rref(self, iszerofunc=None, simplify=False, pivots=True, normalize_last=True):
        """Checked rational reduction, with the profile's dense final result.

        The elimination stays sparse. SymPy 1.14 returns a mutable dense RREF
        for rational sparse inputs, even when the input is immutable; only
        that final representation is subject to the dense storage bound.
        """
        _reduction_options(iszerofunc, simplify)
        if type(pivots) is not bool or type(normalize_last) is not bool:
            raise TypeError("pivots and normalize_last must be bool values")
        checked = _checked_sparse_reduction(self, "rref")
        if checked is not None:
            reduced, pivot_columns = checked
            result = _checked_sparse_lift(reduced, Matrix)
        else:
            reduced, pivot_columns = self.to_dense().rref(
                simplify=simplify, normalize_last=normalize_last,
            )
            # Nonrational inputs use the existing bounded symbolic lane and
            # retain their sparse mutable/immutable class, as in the profile.
            result = self._new(reduced.to_sparse())
        return (result, tuple(pivot_columns)) if pivots else result

    def rank(self, iszerofunc=None, simplify=False):
        """Return exact rational rank without materializing implicit zeros."""
        _reduction_options(iszerofunc, simplify)
        checked = _checked_sparse_reduction(self, "rank")
        return self.to_dense().rank() if checked is None else checked

    def nullspace(self, simplify=False, iszerofunc=None):
        """Return a complete sparse basis in increasing free-column order."""
        _reduction_options(iszerofunc, simplify)
        checked = _checked_sparse_reduction(self, "nullspace")
        if checked is not None:
            return [_checked_sparse_lift(vector, type(self)) for vector in checked]
        return [self._new(vector.to_sparse()) for vector in self.to_dense().nullspace()]

    def columnspace(self, simplify=False):
        """Return original sparse pivot columns, retaining this matrix class."""
        _reduction_options(simplify=simplify)
        checked = _checked_sparse_reduction(self, "columnspace")
        if checked is not None:
            return [_checked_sparse_lift(vector, type(self)) for vector in checked]
        return [self._new(vector.to_sparse()) for vector in self.to_dense().columnspace()]

    colspace = columnspace

    def col(self, j: int) -> "SparseMatrix":
        return self.to_dense().col(j).to_sparse()

    def row(self, i: int) -> "SparseMatrix":
        return self.to_dense().row(i).to_sparse()

    def col_join(self, other: Any) -> "SparseMatrix":
        rhs = other.to_dense() if hasattr(other, "to_dense") else other
        return self.to_dense().col_join(rhs).to_sparse()

    def row_join(self, other: Any) -> "SparseMatrix":
        rhs = other.to_dense() if hasattr(other, "to_dense") else other
        return self.to_dense().row_join(rhs).to_sparse()

    def extract(self, rowsList: Any, colsList: Any) -> "SparseMatrix":
        return self.to_dense().extract(rowsList, colsList).to_sparse()

    def copy(self) -> "SparseMatrix":
        return self._new(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return type(self), (self.rows, self.cols, self.todok())

    def evalf(self, n: int = 15, **options: Any) -> "SparseMatrix":
        from ..core import N

        return self.applyfunc(lambda value: N(value, n, **options))

    n = evalf

    @property
    def free_symbols(self) -> Any:
        sources = _capture_sources(self)
        symbols = set()
        for value in self._operation_entries().values():
            symbols.update(value.free_symbols)
            _check_sources(sources)
        return symbols

    def subs(self, *args: Any, **kwargs: Any) -> "SparseMatrix":
        return self.applyfunc(lambda value: value.subs(*args, **kwargs))

    def applyfunc(self, f: Any) -> "SparseMatrix":
        """Apply f to stored cells in insertion order, keeping implicit zeros.

        This follows the sparse callback contract even when f(0) is nonzero:
        no invocation is made for an absent coordinate.
        """
        if not callable(f):
            raise TypeError("applyfunc requires a callable")
        sources = _capture_sources(self)
        rows, cols = self.shape
        entries = self._operation_entries()
        result = {}
        for coordinate, value in entries.items():
            result[coordinate] = _checked_entry(f(value), sources)
        return self._finish_operation(rows, cols, result, sources)

    def simplify(self) -> "SparseMatrix":
        from ..core import simplify

        return self.applyfunc(simplify)

    def diff(self, *args: Any) -> "SparseMatrix":
        from ..core import diff

        return self.applyfunc(lambda value: diff(value, *args))

    def integrate(self, *args: Any) -> "SparseMatrix":
        from ..integrals import integrate

        return self.applyfunc(lambda value: integrate(value, *args))

    def jacobian(self, X) -> "SparseMatrix":
        """Differentiate a vector using stored cells and supervised scalar rules."""
        from ..core import diff

        if self.rows != 1 and self.cols != 1:
            raise TypeError("self must be a row or a column matrix")
        sources = _capture_sources(self)
        if isinstance(X, (Matrix, SparseMatrix)):
            xrows, xcols = X.shape
            sources += _capture_sources(X)
            if xrows != 1 and xcols != 1:
                raise TypeError("X must be a row or a column matrix")
            variable_count = xrows*xcols
            if variable_count > _MAX_SPARSE_ENTRIES:
                raise ValueError("sparse Jacobian variable count exceeds the stored-entry limit")
            if (self.nnz + 1)*variable_count > _MAX_SPARSE_PRODUCTS:
                raise ValueError("sparse Jacobian exceeds the differentiation work limit")
            if isinstance(X, SparseMatrix):
                values = X._operation_entries()
                raw_variables = tuple(values.get(divmod(index, xcols), _ZERO)
                                      for index in range(variable_count))
            elif X._surface is not None:
                raw_variables = tuple(X._surface)
            else:
                native_values = tuple(X._native_storage[index]
                                      for index in range(variable_count))
                for value in native_values:
                    _admit_native_cell(value)
                raw_variables = tuple(_wrap(value) for value in native_values)
        elif isinstance(X, (list, tuple)):
            if not X:
                raise TypeError("X must be a row or a column matrix")
            if isinstance(X[0], (list, tuple)):
                xrows, xcols = len(X), len(X[0])
                if xrows != 1 and xcols != 1:
                    raise TypeError("X must be a row or a column matrix")
                variable_count = xrows*xcols
                if variable_count > _MAX_SPARSE_ENTRIES:
                    raise ValueError("sparse Jacobian variable count exceeds the stored-entry limit")
                if any(not isinstance(row, (list, tuple)) or len(row) != xcols for row in X):
                    raise ValueError("Row length mismatch in Jacobian variables")
                raw_variables = tuple(value for row in X for value in row)
            else:
                variable_count = len(X)
                if variable_count > _MAX_SPARSE_ENTRIES:
                    raise ValueError("sparse Jacobian variable count exceeds the stored-entry limit")
                raw_variables = tuple(X)
            if (self.nnz + 1)*variable_count > _MAX_SPARSE_PRODUCTS:
                raise ValueError("sparse Jacobian exceeds the differentiation work limit")
        else:
            raise TypeError("X must be a row or a column matrix")

        variables = tuple(_checked_entry(value, sources) for value in raw_variables)
        # Implicit zero residuals still require valid differentiation targets.
        # Use the scalar protocol, including Function/Derivative coordinates,
        # even when this sparse vector has no stored residual expressions.
        for variable in variables:
            _checked_entry(diff(_ZERO, variable), sources)
        rows, cols = _sparse_dimensions(self.rows*self.cols, len(variables))
        entries = self._operation_entries()
        result = {}
        # Jacobian construction visits logical residual rows, unlike applyfunc's
        # insertion-order traversal. Missing residuals have only zero partials.
        for (row, col), value in sorted(entries.items()):
            output_row = row*self.cols + col
            for output_col, variable in enumerate(variables):
                derivative = _checked_entry(diff(value, variable), sources)
                if not _is_exact_zero(derivative):
                    result[(output_row, output_col)] = derivative
                    _stored_entries(result)
        return self._finish_operation(rows, cols, result, sources)

    def todok(self) -> Dict[Tuple[int, int], Any]:
        """Copy stored coordinates only, preserving each actual surface object."""
        if self._surface is not None:
            return dict(self._surface)
        return {coordinate: _wrap(value) for coordinate, value in self._native_storage.todok().items()}

    def __len__(self) -> int:
        return self.rows*self.cols

    def _coordinate(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            row, col = map(int, key)
            if row < 0:
                row += self.rows
            if col < 0:
                col += self.cols
        elif isinstance(key, int):
            size = self.rows*self.cols
            if key < 0:
                key += size
            if not 0 <= key < size:
                raise IndexError("Matrix index out of bounds")
            row, col = divmod(key, self.cols)
        else:
            raise TypeError("SparseMatrix index must be an integer or (row, col) integer pair")
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix index out of bounds")
        return row, col

    def __getitem__(self, key: Any) -> Any:
        coordinate = self._coordinate(key)
        if self._surface is not None:
            return self._surface.get(coordinate, _ZERO)
        return _wrap(self._native_storage[coordinate])

    def __setitem__(self, key: Any, value: Any) -> None:
        sources = _capture_sources(self)
        coordinate = self._coordinate(key)
        entries = self._operation_entries()
        entries[coordinate] = value
        self._install_surface(*self.shape, entries, mutate=True, expected_sources=sources)

    def _native_algebra(self, other):
        if self._native_storage is None or other._native_storage is None:
            return False
        # Preserve native-only provenance and the established exact rational
        # fast paths. Other retained cells keep their original surface objects.
        return (self._surface is None and other._surface is None
                or self._native_storage.is_exact_rational
                and other._native_storage.is_exact_rational)

    def _combine(self, other, subtract=False):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        cls = self._algebra_class(other)
        if self._native_algebra(other):
            result = (self._native_storage - other._native_storage if subtract
                      else self._native_storage + other._native_storage)
            return cls(result)
        sources = _capture_sources(self, other)
        left, right = self._operation_entries(), other._operation_entries()
        _stored_entries(left.keys() | right.keys())
        result = dict(left)
        for coordinate, value in right.items():
            if coordinate in left:
                value = left[coordinate] - value if subtract else left[coordinate] + value
            elif subtract:
                value = -value
            result[coordinate] = _checked_entry(value, sources)
        return self._finish_operation(self.rows, self.cols, result, sources, cls)

    def __add__(self, other: Any) -> "SparseMatrix":
        if isinstance(other, SparseMatrix):
            return self._combine(other)
        if isinstance(other, Matrix):
            return self.to_dense() + other
        raise TypeError(f"Cannot add SparseMatrix and {type(other)}")

    def __sub__(self, other: Any) -> "SparseMatrix":
        if isinstance(other, SparseMatrix):
            return self._combine(other, subtract=True)
        if isinstance(other, Matrix):
            return self.to_dense() - other
        raise TypeError(f"Cannot subtract {type(other)} from SparseMatrix")

    def __neg__(self) -> "SparseMatrix":
        return self * -1

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, Matrix):
            return self.to_dense() @ other
        if not isinstance(other, SparseMatrix):
            raise TypeError(f"Cannot matmul SparseMatrix and {type(other)}")
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        cls = self._algebra_class(other)
        if self._native_algebra(other):
            return cls(self._native_storage @ other._native_storage)
        sources = _capture_sources(self, other)
        rows, cols = _sparse_dimensions(self.rows, other.cols)
        left, right = self._operation_entries(), other._operation_entries()
        right_rows = {}
        for (row, col), value in right.items():
            right_rows.setdefault(row, []).append((col, value))
        work = sum(len(right_rows.get(inner, ())) for _, inner in left)
        if work > _MAX_SPARSE_PRODUCTS:
            raise ValueError("sparse multiplication exceeds the term-product limit")

        # Bound both work and potential output fill before any scalar operator.
        terms = {}
        for row, inner in left:
            for col, _ in right_rows.get(inner, ()):
                terms.setdefault((row, col), [])
                _stored_entries(terms)
        for (row, inner), left_value in left.items():
            for col, right_value in right_rows.get(inner, ()):
                terms[(row, col)].append(_checked_entry(left_value*right_value, sources))

        from ..core import Add

        result = {coordinate: _checked_entry(Add(*values), sources)
                  for coordinate, values in terms.items()}
        return self._finish_operation(rows, cols, result, sources, cls)

    def _scale(self, other, *, left=False):
        from ..core._opaque_arithmetic import contains_opaque

        sources = _capture_sources(self)
        scalar = _checked_entry(other, sources)
        if not contains_opaque(scalar):
            native_scalar = _native_expr(scalar)
            if (self._native_storage is not None
                    and (self._surface is None
                         or self._native_storage.is_exact_rational
                         and (native_scalar.is_integer or native_scalar.is_rational))):
                return self._new(self._native_storage*native_scalar)
            if self.nnz < self.rows*self.cols:
                # Preserve the native refusal for indeterminate implicit
                # zero products such as 0*oo, with a constant-size probe.
                _NativeSparseMatrix.zeros(1, 1)*native_scalar
        result = {}
        for coordinate, value in self._operation_entries().items():
            product = scalar*value if left else value*scalar
            result[coordinate] = _checked_entry(product, sources)
        return self._finish_operation(self.rows, self.cols, result, sources)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (SparseMatrix, Matrix)):
            return self.__matmul__(other)
        if isinstance(other, (Expr, Basic, int, float)):
            return self._scale(other)
        raise TypeError(f"Cannot multiply SparseMatrix and {type(other)}")

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Expr, Basic, int, float)):
            return self._scale(other, left=True)
        if isinstance(other, (SparseMatrix, Matrix)):
            return other.__matmul__(self)
        raise TypeError(f"Cannot multiply {type(other)} and SparseMatrix")

    @classmethod
    def eye(cls, n: int) -> "SparseMatrix":
        return cls(_NativeSparseMatrix.eye(int(n)))

    @classmethod
    def zeros(cls, r: int, c: Any = None) -> "SparseMatrix":
        if c is None:
            c = r
        rows, cols = _sparse_dimensions(r, c)
        return cls(_NativeSparseMatrix.zeros(rows, cols))

    @classmethod
    def ones(cls, r: int, c: Any = None) -> "SparseMatrix":
        if c is None:
            c = r
        rows, cols = _sparse_dimensions(r, c)
        if rows*cols > _MAX_SPARSE_ENTRIES:
            raise ValueError("sparse matrix exceeds the stored-entry limit")
        return cls(rows, cols, [1] * (rows*cols))

    def __repr__(self) -> str:
        return f"SparseMatrix({self.rows}, {self.cols}, {self.todok()})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SparseMatrix):
            if self.shape != other.shape:
                return False
            left, right = self.todok(), other.todok()
            return all(left.get(key, _ZERO) == right.get(key, _ZERO) for key in left.keys() | right.keys())
        if isinstance(other, MatrixBase):
            if hasattr(other, "to_dense"):
                return self.to_dense() == other.to_dense()
            return self.to_dense() == other
        return False


class ImmutableSparseMatrix(SparseMatrix):
    """Immutable sparse snapshot retaining the original symbolic objects."""

    def __init__(self, *args):
        if hasattr(self, "_matrix_shape"):
            raise TypeError("Cannot reinitialize an immutable matrix")
        super().__init__(*args)

    def __setitem__(self, key, value):
        raise TypeError("Cannot set an item on an immutable matrix")

    def __hash__(self):
        entries = self._native_storage.todok() if self._native_storage is not None else self._surface
        return hash((self.shape, tuple((r, c, value) for (r, c), value in sorted(entries.items()))))

    def copy(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def as_immutable(self):
        return self


MutableSparseMatrix = SparseMatrix

__all__ = ["MutableSparseMatrix", "SparseMatrix", "ImmutableSparseMatrix"]

# Capture complete result layouts after both sparse classes exist. Checked
# results bypass _new and constructor dispatch; changed construction hooks
# are refused instead of being silently invoked or discarded.
_REDUCTION_RESULT_LAYOUTS = tuple(
    (cls, base,
     tuple(_REDUCTION_DICT.__get__(base)[name] for name in _REDUCTION_STORAGE_NAMES),
     tuple((name, _reduction_binding(cls, name))
           for name in ("__new__", "__init__", "__setattr__", "__delattr__", "_native")))
    for cls, base in (
        (Matrix, Matrix), (ImmutableDenseMatrix, Matrix),
        (SparseMatrix, SparseMatrix), (ImmutableSparseMatrix, SparseMatrix),
    )
)

_AFFINE_SYMBOL_INTERFACES = tuple(
    (name, _reduction_binding(Symbol, name))
    for name in ("__new__", "__init__", "__setattr__", "__delattr__", "_value", "_assumptions")
)
