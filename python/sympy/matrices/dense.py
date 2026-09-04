"""Dense matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from ..core import Basic, Expr, _native, _native_expr, _wrap

_NativeMatrix = _native.Matrix


class MatrixBase:
    """Base class for all matrix objects."""

    pass


class Matrix(MatrixBase):
    """Exact matrix with SymPy-compatible API wrapping the native linear algebra engine."""

    def __init__(self, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _NativeMatrix):
                self._native = arg
                return
            if isinstance(arg, Matrix):
                self._native = arg._native
                return
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    self._native = _NativeMatrix(0, 0, [])
                    return
                if isinstance(arg[0], (list, tuple)):
                    rows = len(arg)
                    cols = len(arg[0])
                    flat = []
                    for row in arg:
                        if len(row) != cols:
                            raise ValueError("Row length mismatch in matrix constructor")
                        for elem in row:
                            flat.append(_native_expr(elem))
                    self._native = _NativeMatrix(rows, cols, flat)
                    return
                else:
                    # 1D sequence -> column vector
                    rows = len(arg)
                    cols = 1
                    flat = [_native_expr(elem) for elem in arg]
                    self._native = _NativeMatrix(rows, cols, flat)
                    return
            raise TypeError(f"Cannot construct Matrix from {type(arg)}")
        elif len(args) == 2:
            rows, cols = args
            self._native = _NativeMatrix.zeros(int(rows), int(cols))
            return
        elif len(args) == 3:
            rows, cols, entries = args
            rows = int(rows)
            cols = int(cols)
            if callable(entries):
                flat = []
                for r in range(rows):
                    for c in range(cols):
                        flat.append(_native_expr(entries(r, c)))
                self._native = _NativeMatrix(rows, cols, flat)
                return
            elif isinstance(entries, (list, tuple)):
                flat = [_native_expr(elem) for elem in entries]
                self._native = _NativeMatrix(rows, cols, flat)
                return
            else:
                raise TypeError("Third argument to Matrix must be a sequence or callable")
        else:
            raise TypeError(f"Matrix constructor takes 1, 2, or 3 arguments, got {len(args)}")

    @property
    def shape(self):
        return self._native.shape

    @property
    def rows(self):
        return self._native.rows

    @property
    def cols(self):
        return self._native.cols

    @property
    def is_square(self):
        return self._native.is_square

    @property
    def is_symmetric(self):
        return self._native.is_symmetric

    @property
    def is_diagonal(self):
        return self._native.is_diagonal

    @property
    def is_upper_triangular(self):
        return self._native.is_upper_triangular

    @property
    def is_lower_triangular(self):
        return self._native.is_lower_triangular

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return Matrix(self._native.transpose())

    def trace(self):
        return _wrap(self._native.trace())

    def det(self):
        return _wrap(self._native.det())

    def inv(self):
        return Matrix(self._native.inv())

    def inverse(self):
        return self.inv()

    def adjugate(self):
        return Matrix(self._native.adjugate())

    def cofactor(self, r, c):
        return _wrap(self._native.cofactor(int(r), int(c)))

    def frobenius_norm_squared(self):
        return _wrap(self._native.frobenius_norm_squared())

    def rank(self):
        return self._native.rank()

    def rref(self):
        m, pivots = self._native.rref()
        return Matrix(m), tuple(pivots)

    def nullspace(self):
        bases = self._native.nullspace()
        return [Matrix(b) for b in bases]

    def eigenvalues(self):
        return [_wrap(e) for e in self._native.eigenvalues()]

    def eigenvals(self):
        """Return a dict mapping eigenvalue -> multiplicity."""
        evals = self.eigenvalues()
        res = {}
        for ev in evals:
            res[ev] = res.get(ev, 0) + 1
        return res

    def tolist(self):
        return [[_wrap(elem) for elem in row] for row in self._native.to_list()]

    def __len__(self):
        return len(self._native)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            r, c = key
            return _wrap(self._native[(int(r), int(c))])
        if isinstance(key, int):
            return _wrap(self._native[key])
        raise TypeError("Matrix index must be an integer or (row, col) integer pair")

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot add Matrix and {type(other)}")
        return Matrix(self._native + other._native)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot subtract {type(other)} from Matrix")
        return Matrix(self._native - other._native)

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot matmul Matrix and {type(other)}")
        return Matrix(self._native @ other._native)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self._native @ other._native)
        if isinstance(other, (Expr, int)):
            return Matrix(self._native * _native_expr(other))
        raise TypeError(f"Cannot multiply Matrix and {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise TypeError("Matrix power only supports non-negative integers")
        return Matrix(self._native ** n)

    def __repr__(self):
        return f"Matrix({self.tolist()})"

    def __str__(self):
        return str(self._native)

    def _repr_latex_(self):
        return self._native._repr_latex_()

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        return self._native.flat() == other._native.flat()


DenseMatrix = Matrix
MutableDenseMatrix = Matrix


class ImmutableDenseMatrix(Matrix):
    """Hashable immutable dense matrix (SymPy 1.14 semantics, finding 13).

    Upstream immutable matrices are hashable content-addressed values; the
    mutable base stays unhashable exactly as in modern SymPy.
    """

    def __hash__(self):
        return hash((self.shape, tuple(self._native.flat())))


ImmutableMatrix = ImmutableDenseMatrix


def eye(n):
    return Matrix(_NativeMatrix.eye(int(n)))


def zeros(r, c=None):
    if c is None:
        c = r
    return Matrix(_NativeMatrix.zeros(int(r), int(c)))


def diag(*entries):
    flat = []
    for elem in entries:
        if isinstance(elem, (list, tuple)):
            for sub in elem:
                flat.append(_native_expr(sub))
        else:
            flat.append(_native_expr(elem))
    return Matrix(_NativeMatrix.diag(flat))
