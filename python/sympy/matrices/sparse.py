"""Sparse matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple
from ..core import Basic, Expr, Integer, _native, _native_expr, _wrap
from .dense import Matrix, MatrixBase, ImmutableDenseMatrix, _matrix_dimensions, _surface_entry

_NativeSparseMatrix = _native.SparseMatrix


# Sparse dimensions do not impose a dense-area bound. Only explicit cells and
# operations which actually enumerate that area carry storage/work limits.
_MAX_SPARSE_ENTRIES = 262_144
_MAX_SPARSE_PRODUCTS = 1_000_000
_ZERO = Integer(0)


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

    def _install_surface(self, rows, cols, entries, *, mutate=False, expected_generation=None):
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
            value = _surface_entry(value)
            # Never ask a custom expression's equality/printing/assumptions
            # hooks whether it is zero. A held or unknown zero stays retained.
            if _is_exact_zero(value):
                continue
            opaque = contains_opaque(value) or opaque
            retained[(row, col)] = value
        native = None
        if not opaque:
            native = _NativeSparseMatrix(rows, cols, {
                coordinate: _native_expr(value) for coordinate, value in retained.items()
            })
        if expected_generation is not None and self._mutation_generation != expected_generation:
            raise RuntimeError("SparseMatrix mutated during callback traversal")
        self._native_storage = native
        self._matrix_shape = (rows, cols)
        self._surface = retained
        if mutate:
            self._mutation_generation += 1

    def _new(self, *args):
        return type(self)(*args)

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
        return _wrap(self._native.trace())

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

    def det(self, method: Any = None) -> Any:
        if self._native.is_exact_rational:
            return _wrap(self._native.det())
        return self.to_dense().det(method=method)

    def inv(self, method: Any = None, **kwargs: Any) -> "SparseMatrix":
        if self._native.is_exact_rational:
            return self.solve(SparseMatrix.eye(self.rows))
        return self.to_dense().inv(method=method, **kwargs).to_sparse()

    def solve(self, b: Any) -> "SparseMatrix":
        """Solve an exact rational sparse system without dense conversion."""
        if isinstance(b, SparseMatrix):
            sparse_rhs = b
        elif isinstance(b, Matrix):
            sparse_rhs = b.to_sparse()
        elif not hasattr(b, "to_dense"):
            sparse_rhs = SparseMatrix(Matrix(b))
        else:
            sparse_rhs = SparseMatrix(b.to_dense())
        if self._native.is_exact_rational and sparse_rhs._native.is_exact_rational:
            return SparseMatrix(self._native.solve(sparse_rhs._native))
        return self.to_dense().solve(sparse_rhs.to_dense()).to_sparse()

    def LUsolve(self, b: Any) -> "SparseMatrix":
        return self.solve(b)

    def rref(self, **kwargs: Any) -> Any:
        res = self.to_dense().rref(**kwargs)
        if isinstance(res, tuple):
            return res[0].to_sparse(), res[1]
        return res.to_sparse()

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
        return self.to_dense().evalf(n, **options).to_sparse()

    n = evalf

    @property
    def free_symbols(self) -> Any:
        return self.to_dense().free_symbols

    def subs(self, *args: Any, **kwargs: Any) -> "SparseMatrix":
        return self.to_dense().subs(*args, **kwargs).to_sparse()

    def applyfunc(self, f: Any) -> "SparseMatrix":
        return self.to_dense().applyfunc(f).to_sparse()

    def simplify(self) -> "SparseMatrix":
        return self.to_dense().simplify().to_sparse()

    def diff(self, *args: Any) -> "SparseMatrix":
        return self.to_dense().diff(*args).to_sparse()

    def integrate(self, *args: Any) -> "SparseMatrix":
        return self.to_dense().integrate(*args).to_sparse()

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
        generation = self._mutation_generation
        coordinate = self._coordinate(key)
        entries = self.todok()
        entries[coordinate] = value
        self._install_surface(*self.shape, entries, mutate=True, expected_generation=generation)

    def __add__(self, other: Any) -> "SparseMatrix":
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self._native + other._native)
        if isinstance(other, Matrix):
            return self.to_dense() + other
        raise TypeError(f"Cannot add SparseMatrix and {type(other)}")

    def __sub__(self, other: Any) -> "SparseMatrix":
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self._native - other._native)
        if isinstance(other, Matrix):
            return self.to_dense() - other
        raise TypeError(f"Cannot subtract {type(other)} from SparseMatrix")

    def __neg__(self) -> "SparseMatrix":
        return self * -1

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self._native @ other._native)
        if isinstance(other, Matrix):
            return self.to_dense() @ other
        raise TypeError(f"Cannot matmul SparseMatrix and {type(other)}")

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self._native @ other._native)
        if isinstance(other, Matrix):
            return self.to_dense() @ other
        if isinstance(other, (Expr, Basic, int, float)):
            return SparseMatrix(self._native * _native_expr(other))
        raise TypeError(f"Cannot multiply SparseMatrix and {type(other)}")

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

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
