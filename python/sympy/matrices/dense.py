"""Dense matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from ..core import Basic, Expr, Rational, _native, _native_expr, _wrap

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

    def col(self, j):
        cols = self.cols
        rows = self.rows
        j = int(j)
        if j < 0:
            j += cols
        if j < 0 or j >= cols:
            raise IndexError(f"Column index {j} out of range (cols={cols})")
        return Matrix([[self[r, j]] for r in range(rows)])

    def row(self, i):
        rows = self.rows
        cols = self.cols
        i = int(i)
        if i < 0:
            i += rows
        if i < 0 or i >= rows:
            raise IndexError(f"Row index {i} out of range (rows={rows})")
        return Matrix([[self[i, c] for c in range(cols)]])

    def col_join(self, other):
        if not isinstance(other, MatrixBase):
            raise TypeError(f"Cannot col_join Matrix with {type(other)}")
        if self.cols != other.cols:
            raise ValueError(f"Shape mismatch for col_join: {self.cols} vs {other.cols} columns")
        r1, c = self.rows, self.cols
        r2 = other.rows
        data = []
        for r in range(r1):
            for col in range(c):
                data.append(_native_expr(self[r, col]))
        for r in range(r2):
            for col in range(c):
                data.append(_native_expr(other[r, col]))
        return Matrix(_NativeMatrix(r1 + r2, c, data))

    def row_join(self, other):
        if not isinstance(other, MatrixBase):
            raise TypeError(f"Cannot row_join Matrix with {type(other)}")
        if self.rows != other.rows:
            raise ValueError(f"Shape mismatch for row_join: {self.rows} vs {other.rows} rows")
        r = self.rows
        c1 = self.cols
        c2 = other.cols
        data = []
        for row in range(r):
            for col in range(c1):
                data.append(_native_expr(self[row, col]))
            for col in range(c2):
                data.append(_native_expr(other[row, col]))
        return Matrix(_NativeMatrix(r, c1 + c2, data))

    @staticmethod
    def hstack(*args):
        if not args:
            return Matrix(0, 0, [])
        res = args[0]
        if not isinstance(res, Matrix):
            res = Matrix(res)
        for m in args[1:]:
            if not isinstance(m, Matrix):
                m = Matrix(m)
            res = res.row_join(m)
        return res

    @staticmethod
    def vstack(*args):
        if not args:
            return Matrix(0, 0, [])
        res = args[0]
        if not isinstance(res, Matrix):
            res = Matrix(res)
        for m in args[1:]:
            if not isinstance(m, Matrix):
                m = Matrix(m)
            res = res.col_join(m)
        return res

    def eigenvalues(self):
        return [_wrap(e) for e in self._native.eigenvalues()]

    def eigenvals(self):
        """Return a dict mapping eigenvalue -> multiplicity."""
        evals = self.eigenvalues()
        res = {}
        for ev in evals:
            res[ev] = res.get(ev, 0) + 1
        return res

    def eigenvects(self):
        """Return eigenvalues, multiplicities, and eigenvectors: [(eval, mult, [evec, ...]), ...]."""
        eval_dict = self.eigenvals()
        n = self.rows
        I = eye(n)
        res = []
        for ev, mult in eval_dict.items():
            M = self - ev * I
            basis = M.nullspace()
            res.append((ev, mult, basis))
        return res

    def diagonalize(self):
        """Diagonalize matrix M = P * D * P^-1 returning (P, D)."""
        if not self.is_square:
            raise ValueError("Only square matrices can be diagonalized")
        vects = self.eigenvects()
        all_evecs = []
        eval_list = []
        for ev, mult, basis in vects:
            for v in basis:
                all_evecs.append(v)
                eval_list.append(ev)
        if len(all_evecs) < self.rows:
            raise ValueError("Matrix is not diagonalizable (insufficient eigenvectors)")
        P = Matrix.hstack(*all_evecs)
        D = diag(*eval_list)
        return P, D

    def charpoly(self, x=None):
        """Return the characteristic polynomial of this square matrix."""
        if x is None:
            from ..core import Symbol
            x = Symbol("lambda")
        elif isinstance(x, str):
            from ..core import Symbol
            x = Symbol(x)
        coeffs = self._native.char_poly()
        n = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = n - i
            c_expr = _wrap(c)
            if power == 0:
                terms.append(c_expr)
            elif power == 1:
                terms.append(c_expr * x)
            else:
                terms.append(c_expr * (x ** power))
        from ..core import Add
        return Add(*terms)

    def kron(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot compute Kronecker product with {type(other)}")
        return Matrix(self._native.kron(other._native))

    def kronecker_product(self, other):
        return self.kron(other)

    def hadamard(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot compute Hadamard product with {type(other)}")
        return Matrix(self._native.hadamard(other._native))

    def LUdecomposition(self):
        p, l, u = self._native.lu()
        return Matrix(l), Matrix(u), Matrix(p)

    def lu(self):
        p, l, u = self._native.lu()
        return Matrix(p), Matrix(l), Matrix(u)

    def QRdecomposition(self):
        q, r = self._native.qr()
        return Matrix(q), Matrix(r)

    def qr(self):
        return self.QRdecomposition()

    def solve(self, b):
        if not isinstance(b, Matrix):
            raise TypeError(f"solve requires Matrix right-hand side, got {type(b)}")
        return Matrix(self._native.solve(b._native))

    def LUsolve(self, b):
        return self.solve(b)

    def solve_least_squares(self, b):
        if not isinstance(b, Matrix):
            raise TypeError(f"solve_least_squares requires Matrix right-hand side, got {type(b)}")
        return Matrix(self._native.solve_least_squares(b._native))

    def tolist(self):
        return [[_wrap(elem) for elem in row] for row in self._native.to_list()]

    @property
    def free_symbols(self):
        """Return the set of free symbols present in any matrix entry."""
        syms = set()
        for r in range(self.rows):
            for c in range(self.cols):
                val = self[r, c]
                if hasattr(val, "free_symbols"):
                    syms.update(val.free_symbols)
        return syms

    def subs(self, *args, **kwargs):
        """Substitute symbols in all matrix entries."""
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                val = self[r, c]
                if hasattr(val, "subs"):
                    flat.append(_native_expr(val.subs(*args, **kwargs)))
                else:
                    flat.append(_native_expr(val))
        return Matrix(_NativeMatrix(rows, cols, flat))

    def simplify(self):
        """Simplify all matrix entries."""
        from ..core import simplify
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(simplify(self[r, c])))
        return Matrix(_NativeMatrix(rows, cols, flat))

    def diff(self, *args):
        """Differentiate all matrix entries."""
        from ..core import diff
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(diff(self[r, c], *args)))
        return Matrix(_NativeMatrix(rows, cols, flat))

    def integrate(self, *args):
        """Integrate all matrix entries."""
        from ..integrals import integrate
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(integrate(self[r, c], *args)))
        return Matrix(_NativeMatrix(rows, cols, flat))

    def applyfunc(self, f):
        """Apply a function elementwise to every matrix entry."""
        rows = self.rows
        cols = self.cols
        flat = [_native_expr(f(self[r, c])) for r in range(rows) for c in range(cols)]
        return Matrix(_NativeMatrix(rows, cols, flat))

    def jacobian(self, X):
        """Compute the Jacobian matrix with respect to coordinates X."""
        from ..core import diff
        if isinstance(X, MatrixBase):
            vars_list = [X[i] for i in range(len(X))]
        elif isinstance(X, (list, tuple)):
            vars_list = list(X)
        else:
            vars_list = [X]
        funcs = [self[i] for i in range(len(self))]
        rows = len(funcs)
        cols = len(vars_list)
        flat = []
        for f in funcs:
            for v in vars_list:
                flat.append(_native_expr(diff(f, v)))
        return Matrix(_NativeMatrix(rows, cols, flat))

    def to_sparse(self):
        """Convert to a SparseMatrix."""
        from .sparse import SparseMatrix
        return SparseMatrix(self)

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

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            r, c = key
        elif isinstance(key, int):
            r = key // self.cols
            c = key % self.cols
        else:
            raise TypeError("Matrix index must be an integer or (row, col) integer pair")
        r, c = int(r), int(c)
        if r < 0:
            r += self.rows
        if c < 0:
            c += self.cols
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError("Matrix index out of bounds")
        flat = []
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == (r, c):
                    flat.append(_native_expr(value))
                else:
                    flat.append(_native_expr(self[i, j]))
        self._native = _NativeMatrix(self.rows, self.cols, flat)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot add Matrix and {type(other)}")
        return Matrix(self._native + other._native)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot subtract {type(other)} from Matrix")
        return Matrix(self._native - other._native)

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot matmul Matrix and {type(other)}")
        return Matrix(self._native @ other._native)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self._native @ other._native)
        if isinstance(other, (Expr, Basic, int, float)):
            return Matrix(self._native * _native_expr(other))
        raise TypeError(f"Cannot multiply Matrix and {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int):
            return self * Rational(1, other)
        if isinstance(other, (Expr, Basic, float)):
            return self * (other ** -1)
        raise TypeError(f"Cannot divide Matrix by {type(other)}")

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Matrix power only supports integers")
        if n < 0:
            return Matrix(self.inv()._native ** (-n))
        return Matrix(self._native ** n)

    def __repr__(self):
        return f"Matrix({self.tolist()})"

    def __str__(self):
        # Oracle parity: str(Matrix(...)) is the single-line form (finding 6,
        # printer-parity bead qxr); the native multi-line pretty is not str.
        return f"Matrix({self.tolist()})"

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


def hadamard_product(a, b):
    if not isinstance(a, Matrix) or not isinstance(b, Matrix):
        raise TypeError("hadamard_product requires Matrix arguments")
    return a.hadamard(b)


matrix_multiply_elementwise = hadamard_product


def kronecker_product(a, b):
    if not isinstance(a, Matrix) or not isinstance(b, Matrix):
        raise TypeError("kronecker_product requires Matrix arguments")
    return a.kron(b)


def ones(r, c=None):
    if c is None:
        c = r
    r = int(r)
    c = int(c)
    return Matrix(r, c, [1] * (r * c))


def hstack(*args):
    return Matrix.hstack(*args)


def vstack(*args):
    return Matrix.vstack(*args)


def jacobian(exprs, vars):
    """Compute the Jacobian matrix of expressions with respect to variables."""
    if not isinstance(exprs, MatrixBase):
        exprs = Matrix(exprs)
    return exprs.jacobian(vars)


def wronskian(functions, var):
    """Compute the Wronskian determinant of functions with respect to var."""
    from ..core import diff
    n = len(functions)
    if n == 0:
        return 1
    rows = []
    curr = list(functions)
    rows.append(curr)
    for _ in range(1, n):
        curr = [diff(f, var) for f in curr]
        rows.append(curr)
    return Matrix(rows).det()


def casoratian(seqs, n):
    """Compute the Casoratian determinant of sequences with respect to index n."""
    k = len(seqs)
    if k == 0:
        return 1
    rows = []
    for i in range(k):
        rows.append([s.subs(n, n + i) if hasattr(s, "subs") else s for s in seqs])
    return Matrix(rows).det()


def GramSchmidt(vlist, orthonormal=False):
    """Apply the Gram-Schmidt orthogonalization process to a list of vectors."""
    out = []
    for v in vlist:
        if not isinstance(v, MatrixBase):
            v = Matrix(v)
        w = v
        for u in out:
            # projection of v onto u: (u.T * v)[0, 0] / (u.T * u)[0, 0] * u
            numerator = (u.T * v)[0, 0]
            denominator = (u.T * u)[0, 0]
            w = w - (numerator / denominator) * u
        is_zero_vec = True
        for i in range(len(w)):
            if w[i] != 0:
                is_zero_vec = False
                break
        if not is_zero_vec:
            if orthonormal:
                norm_sq = (w.T * w)[0, 0]
                from ..core import sqrt
                w = w / sqrt(norm_sq)
            out.append(w)
    return out

