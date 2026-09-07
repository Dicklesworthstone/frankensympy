"""Sparse matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from typing import Any, Dict, Tuple
from ..core import Basic, Expr, _native, _native_expr, _wrap
from .dense import Matrix, MatrixBase

_NativeSparseMatrix = _native.SparseMatrix


class SparseMatrix(MatrixBase):
    """Exact sparse matrix wrapping the native compressed coordinate matrix engine."""

    def __init__(self, *args: Any):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _NativeSparseMatrix):
                self._native = arg
                return
            if isinstance(arg, SparseMatrix):
                self._native = arg._native
                return
            if isinstance(arg, Matrix):
                self._native = arg._native.to_sparse()
                return
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    self._native = _NativeSparseMatrix.zeros(0, 0)
                    return
                if isinstance(arg[0], (list, tuple)):
                    rows = len(arg)
                    cols = len(arg[0])
                    entries: Dict[Tuple[int, int], Any] = {}
                    for r, row in enumerate(arg):
                        if len(row) != cols:
                            raise ValueError("Row length mismatch in sparse matrix constructor")
                        for c, elem in enumerate(row):
                            val = _native_expr(elem)
                            if str(val) != "0":
                                entries[(r, c)] = val
                    self._native = _NativeSparseMatrix(rows, cols, entries)
                    return
                else:
                    # 1D column vector
                    rows = len(arg)
                    cols = 1
                    entries = {}
                    for r, elem in enumerate(arg):
                        val = _native_expr(elem)
                        if str(val) != "0":
                            entries[(r, 0)] = val
                    self._native = _NativeSparseMatrix(rows, cols, entries)
                    return
            raise TypeError(f"Cannot construct SparseMatrix from {type(arg)}")
        elif len(args) == 2:
            rows, cols = args
            self._native = _NativeSparseMatrix.zeros(int(rows), int(cols))
            return
        elif len(args) == 3:
            rows, cols, entries_in = args
            rows = int(rows)
            cols = int(cols)
            if isinstance(entries_in, dict):
                entries = {}
                for k, v in entries_in.items():
                    r, c = k
                    entries[(int(r), int(c))] = _native_expr(v)
                self._native = _NativeSparseMatrix(rows, cols, entries)
                return
            elif isinstance(entries_in, (list, tuple)):
                entries = {}
                for idx, elem in enumerate(entries_in):
                    val = _native_expr(elem)
                    if str(val) != "0":
                        r = idx // cols
                        c = idx % cols
                        entries[(r, c)] = val
                self._native = _NativeSparseMatrix(rows, cols, entries)
                return
            elif callable(entries_in):
                entries = {}
                for r in range(rows):
                    for c in range(cols):
                        val = _native_expr(entries_in(r, c))
                        if str(val) != "0":
                            entries[(r, c)] = val
                self._native = _NativeSparseMatrix(rows, cols, entries)
                return
            else:
                raise TypeError("Third argument to SparseMatrix must be a dict, sequence, or callable")
        else:
            raise TypeError(f"SparseMatrix constructor takes 1, 2, or 3 arguments, got {len(args)}")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._native.shape

    @property
    def rows(self) -> int:
        return self._native.rows

    @property
    def cols(self) -> int:
        return self._native.cols

    @property
    def nnz(self) -> int:
        return self._native.nnz

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    @property
    def T(self) -> "SparseMatrix":
        return self.transpose()

    def transpose(self) -> "SparseMatrix":
        return SparseMatrix(self._native.transpose())

    def trace(self) -> Any:
        return _wrap(self._native.trace())

    def to_dense(self) -> Matrix:
        return Matrix(self._native.to_dense())

    def as_mutable(self) -> "SparseMatrix":
        return self

    def as_immutable(self) -> "SparseMatrix":
        return self

    def todok(self) -> Dict[Tuple[int, int], Any]:
        """Return a dictionary of non-zero entries keyed by (row, col)."""
        res = {}
        for r in range(self.rows):
            for c in range(self.cols):
                val = self[r, c]
                if val != 0:
                    res[(r, c)] = val
        return res

    def __len__(self) -> int:
        return len(self._native)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            r, c = key
            return _wrap(self._native[(int(r), int(c))])
        if isinstance(key, int):
            return _wrap(self._native[key])
        raise TypeError("SparseMatrix index must be an integer or (row, col) integer pair")

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            r, c = key
        elif isinstance(key, int):
            r = key // self.cols
            c = key % self.cols
        else:
            raise TypeError("SparseMatrix index must be an integer or (row, col) integer pair")
        r, c = int(r), int(c)
        if r < 0:
            r += self.rows
        if c < 0:
            c += self.cols
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError("Matrix index out of bounds")
        entries = {}
        for (i, j), v in self.todok().items():
            if (i, j) != (r, c):
                entries[(i, j)] = _native_expr(v)
        val = _native_expr(value)
        if str(val) != "0":
            entries[(r, c)] = val
        self._native = _NativeSparseMatrix(self.rows, self.cols, entries)

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

    def __repr__(self) -> str:
        return f"SparseMatrix({self.rows}, {self.cols}, {self.todok()})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SparseMatrix):
            return self._native == other._native
        if isinstance(other, Matrix):
            return self.to_dense() == other
        return False


MutableSparseMatrix = SparseMatrix

__all__ = [
    "MutableSparseMatrix",
    "SparseMatrix",
]
