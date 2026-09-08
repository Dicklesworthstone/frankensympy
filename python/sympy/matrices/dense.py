"""Dense matrix implementations for FrankenSymPy compatibility (WS05, WS10)."""

from ..core import Basic, Expr, Rational, _native, _native_expr, _wrap

_NativeMatrix = _native.Matrix


class _CallableBool(int):
    """Boolean-compatible integer that is also callable as a predicate."""

    def __new__(cls, val):
        return super().__new__(cls, 1 if val else 0)

    def __call__(self, *args, **kwargs):
        return bool(self)


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
            if hasattr(arg, "to_dense") and not isinstance(arg, type):
                self._native = arg.to_dense()._native
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

    def _new(self, *args, **kwargs):
        """Construct a new matrix of the same class (preserving mutability/immutability)."""
        return self.__class__(*args, **kwargs)

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
    def is_zero_matrix(self):
        return _CallableBool(all(self[r, c] == 0 for r in range(self.rows) for c in range(self.cols)))

    @property
    def is_identity(self):
        if not self.is_square:
            return _CallableBool(False)
        for r in range(self.rows):
            for c in range(self.cols):
                expected = 1 if r == c else 0
                if self[r, c] != expected:
                    return _CallableBool(False)
        return _CallableBool(True)

    def is_nilpotent(self):
        if not self.is_square:
            return False
        if self.rows == 0:
            return True
        curr = self
        for _ in range(self.rows):
            if curr.is_zero_matrix:
                return True
            curr = curr @ self
        return bool(curr.is_zero_matrix)

    @property
    def is_symmetric(self):
        return _CallableBool(self._native.is_symmetric)

    @property
    def is_anti_symmetric(self):
        return _CallableBool(self._native.is_skew_symmetric)

    @property
    def is_skew_symmetric(self):
        return self.is_anti_symmetric

    @property
    def is_diagonal(self):
        return _CallableBool(self._native.is_diagonal)

    @property
    def is_upper(self):
        return self._native.is_upper_triangular

    @property
    def is_lower(self):
        return self._native.is_lower_triangular

    @property
    def is_upper_triangular(self):
        return _CallableBool(self._native.is_upper_triangular)

    @property
    def is_lower_triangular(self):
        return _CallableBool(self._native.is_lower_triangular)

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return self._new(self._native.transpose())

    @property
    def H(self):
        """Return the Hermitian transpose (conjugate transpose)."""
        return self.T.conjugate()

    def adjoint(self):
        """Return the Hermitian adjoint (conjugate transpose)."""
        return self.H

    @property
    def C(self):
        """Return the element-wise complex conjugate (SymPy property alias)."""
        return self.conjugate()

    def conjugate(self):
        """Return the element-wise complex conjugate of this matrix."""
        flat = []
        for r in range(self.rows):
            for c in range(self.cols):
                val = self[r, c]
                flat.append(val.conjugate() if hasattr(val, "conjugate") else val)
        return self._new(self.rows, self.cols, flat)

    def is_hermitian(self):
        """Return True if matrix is Hermitian (equal to its conjugate transpose)."""
        if not self.is_square:
            return False
        return self == self.H

    def is_anti_hermitian(self):
        """Return True if matrix is anti-Hermitian (equal to negative conjugate transpose)."""
        if not self.is_square:
            return False
        return self == -self.H

    def trace(self):
        return _wrap(self._native.trace())

    def det(self, method=None):
        return _wrap(self._native.det())

    def inv(self, method=None, **kwargs):
        return self._new(self._native.inv())

    def inverse(self, method=None, **kwargs):
        return self.inv(method=method, **kwargs)

    def adjugate(self):
        return self._new(self._native.adjugate())

    def cofactor(self, r, c):
        return _wrap(self._native.cofactor(int(r), int(c)))

    def cofactor_matrix(self):
        """Return the matrix of cofactors."""
        return self._new(self.rows, self.cols, lambda i, j: self.cofactor(i, j))

    def minor_submatrix(self, i, j):
        """Return the submatrix obtained by deleting row `i` and column `j`."""
        return self._new(self._native.minor_submatrix(int(i), int(j)))

    def minorMatrix(self, i, j):
        """Alias for minor_submatrix."""
        return self.minor_submatrix(i, j)

    def minor(self, i, j, **kwargs):
        """Return the minor (determinant of the minor submatrix)."""
        return self.minor_submatrix(i, j).det()

    def frobenius_norm_squared(self):
        return _wrap(self._native.frobenius_norm_squared())

    def rank(self, iszerofunc=None):
        return self._native.rank()

    def rref(self, iszerofunc=None, simplify=False, pivots=True, normalize_last=True):
        m, pivs = self._native.rref()
        res_m = self._new(m)
        if pivots:
            return res_m, tuple(pivs)
        return res_m

    def nullspace(self):
        bases = self._native.nullspace()
        return [self._new(b) for b in bases]

    def columnspace(self):
        """Return a list of column vectors spanning the column space of this matrix."""
        _, pivots = self.rref()
        return [self.col(p) for p in pivots]

    colspace = columnspace

    def rowspace(self):
        """Return a list of row vectors spanning the row space of this matrix."""
        reduced, pivots = self.rref()
        return [reduced.row(i) for i in range(len(pivots))]

    def col(self, j):
        cols = self.cols
        rows = self.rows
        j = int(j)
        if j < 0:
            j += cols
        if j < 0 or j >= cols:
            raise IndexError(f"Column index {j} out of range (cols={cols})")
        return self._new([[self[r, j]] for r in range(rows)])

    def row(self, i):
        rows = self.rows
        cols = self.cols
        i = int(i)
        if i < 0:
            i += rows
        if i < 0 or i >= rows:
            raise IndexError(f"Row index {i} out of range (rows={rows})")
        return self._new([[self[i, c] for c in range(cols)]])

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
        return self._new(_NativeMatrix(r1 + r2, c, data))

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
        return self._new(_NativeMatrix(r, c1 + c2, data))

    def extract(self, rowsList, colsList):
        """Return a submatrix formed by the given rows and cols indices."""
        r_list = [r if r >= 0 else r + self.rows for r in rowsList]
        c_list = [c if c >= 0 else c + self.cols for c in colsList]
        for r in r_list:
            if not (0 <= r < self.rows):
                raise IndexError(f"Row index {r} out of bounds")
        for c in c_list:
            if not (0 <= c < self.cols):
                raise IndexError(f"Column index {c} out of bounds")
        flat = [_native_expr(self[r, c]) for r in r_list for c in c_list]
        return self._new(_NativeMatrix(len(r_list), len(c_list), flat))

    def col_insert(self, pos, other):
        """Insert a matrix at column pos."""
        if not isinstance(other, Matrix):
            other = Matrix(other)
        if other.rows != self.rows:
            raise ValueError(
                f"Cannot insert column of length {other.rows} into matrix of {self.rows} rows"
            )
        cols = self.cols
        if pos < 0:
            pos += cols
        pos = max(0, min(pos, cols))
        flat = []
        for r in range(self.rows):
            for c in range(pos):
                flat.append(_native_expr(self[r, c]))
            for c in range(other.cols):
                flat.append(_native_expr(other[r, c]))
            for c in range(pos, cols):
                flat.append(_native_expr(self[r, c]))
        return self._new(_NativeMatrix(self.rows, cols + other.cols, flat))

    def row_insert(self, pos, other):
        """Insert a matrix at row pos."""
        if not isinstance(other, Matrix):
            other = Matrix(other)
        if other.cols != self.cols:
            raise ValueError(
                f"Cannot insert row of width {other.cols} into matrix of {self.cols} columns"
            )
        rows = self.rows
        if pos < 0:
            pos += rows
        pos = max(0, min(pos, rows))
        flat = []
        for r in range(pos):
            for c in range(self.cols):
                flat.append(_native_expr(self[r, c]))
        for r in range(other.rows):
            for c in range(self.cols):
                flat.append(_native_expr(other[r, c]))
        for r in range(pos, rows):
            for c in range(self.cols):
                flat.append(_native_expr(self[r, c]))
        return self._new(_NativeMatrix(rows + other.rows, self.cols, flat))

    def col_del(self, j):
        """Delete column j in-place."""
        cols = self.cols
        if j < 0:
            j += cols
        if not (0 <= j < cols):
            raise IndexError("Column index out of range")
        rows = self.rows
        flat = []
        for r in range(rows):
            for c in range(cols):
                if c != j:
                    flat.append(_native_expr(self[r, c]))
        self._native = _NativeMatrix(rows, cols - 1, flat)

    def row_del(self, i):
        """Delete row i in-place."""
        rows = self.rows
        if i < 0:
            i += rows
        if not (0 <= i < rows):
            raise IndexError("Row index out of range")
        cols = self.cols
        flat = []
        for r in range(rows):
            if r != i:
                for c in range(cols):
                    flat.append(_native_expr(self[r, c]))
        self._native = _NativeMatrix(rows - 1, cols, flat)

    def row_swap(self, i, j):
        """Swap row i and row j in-place."""
        rows = self.rows
        cols = self.cols
        i = int(i) if int(i) >= 0 else int(i) + rows
        j = int(j) if int(j) >= 0 else int(j) + rows
        if not (0 <= i < rows and 0 <= j < rows):
            raise IndexError(f"Row indices ({i}, {j}) out of range (rows={rows})")
        if i == j:
            return
        flat = [self[r, c] for r in range(rows) for c in range(cols)]
        for c in range(cols):
            flat[i * cols + c], flat[j * cols + c] = flat[j * cols + c], flat[i * cols + c]
        self._native = _NativeMatrix(rows, cols, [_native_expr(x) for x in flat])

    def col_swap(self, i, j):
        """Swap column i and column j in-place."""
        rows = self.rows
        cols = self.cols
        i = int(i) if int(i) >= 0 else int(i) + cols
        j = int(j) if int(j) >= 0 else int(j) + cols
        if not (0 <= i < cols and 0 <= j < cols):
            raise IndexError(f"Column indices ({i}, {j}) out of range (cols={cols})")
        if i == j:
            return
        flat = [self[r, c] for r in range(rows) for c in range(cols)]
        for r in range(rows):
            flat[r * cols + i], flat[r * cols + j] = flat[r * cols + j], flat[r * cols + i]
        self._native = _NativeMatrix(rows, cols, [_native_expr(x) for x in flat])

    def row_op(self, i, f):
        """In-place apply function f(val, col_index) to each element of row i."""
        rows = self.rows
        cols = self.cols
        i = int(i) if int(i) >= 0 else int(i) + rows
        if not (0 <= i < rows):
            raise IndexError(f"Row index {i} out of range (rows={rows})")
        flat = [self[r, c] for r in range(rows) for c in range(cols)]
        for c in range(cols):
            flat[i * cols + c] = f(flat[i * cols + c], c)
        self._native = _NativeMatrix(rows, cols, [_native_expr(x) for x in flat])

    def col_op(self, j, f):
        """In-place apply function f(val, row_index) to each element of col j."""
        rows = self.rows
        cols = self.cols
        j = int(j) if int(j) >= 0 else int(j) + cols
        if not (0 <= j < cols):
            raise IndexError(f"Column index {j} out of range (cols={cols})")
        flat = [self[r, c] for r in range(rows) for c in range(cols)]
        for r in range(rows):
            flat[r * cols + j] = f(flat[r * cols + j], r)
        self._native = _NativeMatrix(rows, cols, [_native_expr(x) for x in flat])

    def norm(self, ord="fro"):
        """Return the matrix norm (Frobenius / Euclidean by default)."""
        from ..core import sqrt
        if ord in ("fro", 2, None):
            return sqrt(self.frobenius_norm_squared())
        raise NotImplementedError(f"Matrix norm with ord={ord} is not implemented")

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

    def is_diagonalizable(self, reals_only=False):
        """Return True if matrix is diagonalizable (and has real eigenvalues if reals_only=True)."""
        if not self.is_square:
            return False
        if self.rows <= 1:
            if self.rows == 1 and reals_only:
                from ..core import I
                val = self[0, 0]
                if (hasattr(val, "has") and val.has(I)) or getattr(val, "is_real", None) is False:
                    return False
            return True
        from ..core import I
        # Diagonal check
        is_diag = True
        for r in range(self.rows):
            for c in range(self.cols):
                if r != c and self[r, c] != 0:
                    is_diag = False
                    break
            if not is_diag:
                break
        if is_diag:
            if reals_only:
                for i in range(self.rows):
                    val = self[i, i]
                    if (hasattr(val, "has") and val.has(I)) or getattr(val, "is_real", None) is False:
                        return False
            return True

        # Spectral theorem: all Hermitian matrices are diagonalizable over C and have real eigenvalues
        if self.is_hermitian():
            return True

        try:
            vects = self.eigenvects()
            all_evecs = []
            for ev, mult, basis in vects:
                if reals_only:
                    if (hasattr(ev, "has") and ev.has(I)) or getattr(ev, "is_real", None) is False:
                        return False
                all_evecs.extend(basis)
            return len(all_evecs) == self.rows
        except Exception:
            return False

    def exp(self):
        """Return the matrix exponential exp(M)."""
        if not self.is_square:
            raise ValueError("Matrix exponential only defined for square matrices")
        n = self.rows
        if n == 0:
            return self
        if n == 1:
            from ..functions import exp as spexp
            return self._new(1, 1, [spexp(self[0, 0])])
        from ..functions import exp as spexp

        # Diagonal matrix fast path
        is_diag = True
        for r in range(n):
            for c in range(n):
                if r != c and self[r, c] != 0:
                    is_diag = False
                    break
            if not is_diag:
                break
        if is_diag:
            return diag(*[spexp(self[i, i]) for i in range(n)])

        # Diagonalizable path
        try:
            if self.is_diagonalizable():
                P, D = self.diagonalize()
                exp_diag = [spexp(D[i, i]) for i in range(n)]
                return P * diag(*exp_diag) * P.inv()
        except Exception:
            pass

        # Nilpotent or shifted nilpotent expansion: (M - lambda*I)^k = 0
        I_n = eye(n)
        # Check if nilpotent directly
        power = self
        powers = [I_n]
        is_nilpotent = False
        from math import factorial
        for k in range(1, n + 1):
            if power == zeros(n, n):
                is_nilpotent = True
                break
            powers.append(power)
            power = power * self
        if is_nilpotent:
            res = zeros(n, n)
            for k, p_mat in enumerate(powers):
                res = res + p_mat * Rational(1, factorial(k))
            return res

        # Check shifted nilpotent: (M - lambda * I)^k = 0
        try:
            lam = self.trace() / n
            N = self - lam * I_n
            power = N
            powers = [I_n]
            is_shifted_nilpotent = False
            for k in range(1, n + 1):
                if power == zeros(n, n):
                    is_shifted_nilpotent = True
                    break
                powers.append(power)
                power = power * N
            if is_shifted_nilpotent:
                res = zeros(n, n)
                for k, p_mat in enumerate(powers):
                    res = res + p_mat * Rational(1, factorial(k))
                return spexp(lam) * res
        except Exception:
            pass

        raise NotImplementedError("Matrix exponential could not be computed in closed form")

    def is_positive_definite(self):
        """Return True if matrix is Hermitian and positive definite."""
        if not self.is_square:
            return False
        if not self.is_hermitian():
            return False
        n = self.rows
        if n == 0:
            return True
        # Sylvester criterion: all leading principal minors must be strictly positive
        for k in range(1, n + 1):
            sub = self[:k, :k]
            d = sub.det()
            if getattr(d, "is_positive", None) is True:
                continue
            try:
                if d <= 0:
                    return False
            except TypeError:
                return False
        return True

    def is_positive_semidefinite(self):
        """Return True if matrix is Hermitian and positive semidefinite."""
        if not self.is_square:
            return False
        if not self.is_hermitian():
            return False
        n = self.rows
        if n == 0:
            return True
        # Check all diagonal entries >= 0 first
        for i in range(n):
            d = self[i, i]
            if getattr(d, "is_negative", None) is True:
                return False
            try:
                if d < 0:
                    return False
            except TypeError:
                return False
        # If determinant is negative, definitely not PSD
        det_all = self.det()
        try:
            if det_all < 0:
                return False
        except TypeError:
            return False

        # Sylvester-Frobenius: all principal minors must be non-negative
        import itertools
        for k in range(1, n):
            for cols in itertools.combinations(range(n), k):
                cols_list = list(cols)
                sub = self.extract(cols_list, cols_list)
                d = sub.det()
                try:
                    if d < 0:
                        return False
                except TypeError:
                    return False
        return True

    def is_negative_definite(self):
        """Return True if matrix is Hermitian and negative definite."""
        if not self.is_square:
            return False
        return (-self).is_positive_definite()

    def is_negative_semidefinite(self):
        """Return True if matrix is Hermitian and negative semidefinite."""
        if not self.is_square:
            return False
        return (-self).is_positive_semidefinite()

    def singular_values(self):
        """Return the singular values of this matrix in descending order."""
        if self.rows == 0 or self.cols == 0:
            return []
        from ..functions import sqrt
        # Gram matrix: M^H * M (or M * M^H, whichever is smaller)
        if self.rows >= self.cols:
            gram = self.H * self
        else:
            gram = self * self.H

        # If gram is 1x1
        if gram.rows == 1:
            val = gram[0, 0]
            s = sqrt(val) if val >= 0 else sqrt(-val)
            return [s]

        # If gram is diagonal
        is_diag = True
        for r in range(gram.rows):
            for c in range(gram.cols):
                if r != c and gram[r, c] != 0:
                    is_diag = False
                    break
            if not is_diag:
                break
        if is_diag:
            s_vals = [sqrt(gram[i, i]) for i in range(gram.rows)]
            try:
                s_vals.sort(key=lambda s: float(s.evalf()), reverse=True)
            except Exception:
                pass
            return s_vals

        evals = gram.eigenvalues()
        s_vals = []
        for ev in evals:
            s_vals.append(sqrt(ev))
        try:
            s_vals.sort(key=lambda s: float(s.evalf()), reverse=True)
        except Exception:
            pass
        return s_vals

    def condition_number(self):
        """Return the condition number sigma_max / sigma_min."""
        s = self.singular_values()
        if not s:
            return Rational(1)
        s_max = s[0]
        s_min = s[-1]
        if s_min == 0:
            from ..core import zoo
            return zoo
        return s_max / s_min

    def cholesky(self, hermitian=True):
        """Return the lower triangular Cholesky factor L such that L * L.H == self (if hermitian)
        or L * L.T == self (if not hermitian)."""
        from ..functions import sqrt
        if not self.is_square:
            raise ValueError("Matrix must be square.")
        if hermitian and not self.is_hermitian():
            raise ValueError("Matrix must be Hermitian.")
        if not hermitian and not self.is_symmetric():
            raise ValueError("Matrix must be symmetric.")
        n = self.rows
        if n == 0:
            return self
        L = zeros(n, n)
        if hermitian:
            for i in range(n):
                for j in range(i):
                    s = sum(L[i, k] * (L[j, k].conjugate() if hasattr(L[j, k], "conjugate") else L[j, k]) for k in range(j))
                    L[i, j] = (self[i, j] - s) / L[j, j]
                s_diag = sum(L[i, k] * (L[i, k].conjugate() if hasattr(L[i, k], "conjugate") else L[i, k]) for k in range(i))
                rem = self[i, i] - s_diag
                if getattr(rem, "is_positive", None) is False:
                    raise ValueError("Matrix must be positive-definite.")
                try:
                    if rem <= 0:
                        raise ValueError("Matrix must be positive-definite.")
                except TypeError:
                    pass
                L[i, i] = sqrt(rem)
        else:
            for i in range(n):
                for j in range(i):
                    s = sum(L[i, k] * L[j, k] for k in range(j))
                    L[i, j] = (self[i, j] - s) / L[j, j]
                s_diag = sum(L[i, k]**2 for k in range(i))
                rem = self[i, i] - s_diag
                L[i, i] = sqrt(rem)
        return self._new(L)

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
        return self._new(l), self._new(u), self._new(p)

    def lu(self):
        p, l, u = self._native.lu()
        return self._new(p), self._new(l), self._new(u)

    def QRdecomposition(self):
        q, r = self._native.qr()
        return self._new(q), self._new(r)

    def qr(self):
        return self.QRdecomposition()

    def LDLdecomposition(self):
        l, d = self._native.ldl()
        return self._new(l), self._new(d)

    def ldl(self):
        return self.LDLdecomposition()

    def LDLsolve(self, b):
        return self.solve(b)

    def solve(self, b):
        if hasattr(b, "to_dense"):
            b = b.to_dense()
        if not isinstance(b, Matrix):
            raise TypeError(f"solve requires Matrix right-hand side, got {type(b)}")
        return self._new(self._native.solve(b._native))

    def LUsolve(self, b):
        return self.solve(b)

    def solve_least_squares(self, b):
        if hasattr(b, "to_dense"):
            b = b.to_dense()
        if not isinstance(b, Matrix):
            raise TypeError(f"solve_least_squares requires Matrix right-hand side, got {type(b)}")
        return self._new(self._native.solve_least_squares(b._native))

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
        return self._new(_NativeMatrix(rows, cols, flat))

    def simplify(self):
        """Simplify all matrix entries."""
        from ..core import simplify
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(simplify(self[r, c])))
        return self._new(_NativeMatrix(rows, cols, flat))

    def diff(self, *args):
        """Differentiate all matrix entries."""
        from ..core import diff
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(diff(self[r, c], *args)))
        return self._new(_NativeMatrix(rows, cols, flat))

    def integrate(self, *args):
        """Integrate all matrix entries."""
        from ..integrals import integrate
        rows = self.rows
        cols = self.cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append(_native_expr(integrate(self[r, c], *args)))
        return self._new(_NativeMatrix(rows, cols, flat))

    def applyfunc(self, f):
        """Apply a function elementwise to every matrix entry."""
        rows = self.rows
        cols = self.cols
        flat = [_native_expr(f(self[r, c])) for r in range(rows) for c in range(cols)]
        return self._new(_NativeMatrix(rows, cols, flat))

    def evalf(self, n=15, **options):
        """Evaluate all matrix entries numerically."""
        from ..core import N
        rows = self.rows
        cols = self.cols
        flat = [_native_expr(N(self[r, c], n, **options)) for r in range(rows) for c in range(cols)]
        return self._new(_NativeMatrix(rows, cols, flat))

    n = evalf

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
        return self._new(_NativeMatrix(rows, cols, flat))

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
            if isinstance(r, slice) or isinstance(c, slice):
                row_indices = list(range(*r.indices(self.rows))) if isinstance(r, slice) else [int(r) if int(r) >= 0 else int(r) + self.rows]
                col_indices = list(range(*c.indices(self.cols))) if isinstance(c, slice) else [int(c) if int(c) >= 0 else int(c) + self.cols]
                return self.extract(row_indices, col_indices)
            return _wrap(self._native[(int(r), int(c))])
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
            return [self[i] for i in indices]
        if isinstance(key, int):
            return _wrap(self._native[key])
        raise TypeError("Matrix index must be an integer, slice, or (row, col) pair")

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index must be a 2-tuple (row, col)")
            r, c = key
            if isinstance(r, slice) or isinstance(c, slice):
                row_indices = list(range(*r.indices(self.rows))) if isinstance(r, slice) else [int(r) if int(r) >= 0 else int(r) + self.rows]
                col_indices = list(range(*c.indices(self.cols))) if isinstance(c, slice) else [int(c) if int(c) >= 0 else int(c) + self.cols]
                target_len = len(row_indices) * len(col_indices)
                if isinstance(value, MatrixBase):
                    vals = [value[i, j] for i in range(value.rows) for j in range(value.cols)]
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0 and isinstance(value[0], (list, tuple)):
                        vals = [elem for row in value for elem in row]
                    else:
                        vals = list(value)
                else:
                    vals = [value] * target_len
                if len(vals) != target_len:
                    raise ValueError(f"Shape mismatch: cannot assign {len(vals)} values to {len(row_indices)}x{len(col_indices)} submatrix")
                flat = [self[i, j] for i in range(self.rows) for j in range(self.cols)]
                idx = 0
                for ri in row_indices:
                    for ci in col_indices:
                        flat[ri * self.cols + ci] = vals[idx]
                        idx += 1
                self._native = _NativeMatrix(self.rows, self.cols, [_native_expr(x) for x in flat])
                return
            r = int(r)
            c = int(c)
        elif isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
            if not isinstance(value, (list, tuple)):
                value = [value] * len(indices)
            if len(indices) != len(value):
                raise ValueError(f"Shape mismatch: cannot assign {len(value)} values to slice of length {len(indices)}")
            flat = [self[i] for i in range(len(self))]
            for idx, val in zip(indices, value):
                flat[idx] = val
            self._native = _NativeMatrix(self.rows, self.cols, [_native_expr(x) for x in flat])
            return
        elif isinstance(key, int):
            r = key // self.cols
            c = key % self.cols
        else:
            raise TypeError("Matrix index must be an integer, slice, or (row, col) pair")
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
        return self._new(self._native + other._native)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot subtract {type(other)} from Matrix")
        return self._new(self._native - other._native)

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot matmul Matrix and {type(other)}")
        return self._new(self._native @ other._native)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self._new(self._native @ other._native)
        if isinstance(other, (Expr, Basic, int, float)):
            return self._new(self._native * _native_expr(other))
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
            return self._new(self.inv()._native ** (-n))
        return self._new(self._native ** n)

    def copy(self):
        """Return a copy of the matrix."""
        flat = [self[i, j] for i in range(self.rows) for j in range(self.cols)]
        return self._new(_NativeMatrix(self.rows, self.cols, [_native_expr(x) for x in flat]))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def as_immutable(self):
        """Return an immutable copy of the matrix."""
        return ImmutableDenseMatrix(self)

    def as_mutable(self):
        """Return a mutable copy of the matrix."""
        return self.copy()

    def pinv(self):
        """Compute the Moore-Penrose pseudoinverse."""
        if self.rows >= self.cols:
            try:
                return self.solve_least_squares(eye(self.rows))
            except Exception:
                pass
        return self.T.solve_least_squares(eye(self.cols)).T

    def dot(self, other):
        """Compute the inner (dot) product with another matrix/vector."""
        if not isinstance(other, MatrixBase):
            try:
                other = Matrix(other)
            except Exception:
                raise TypeError(f"Cannot compute dot product with {type(other)}")
        if len(self) != len(other):
            raise ValueError(f"Matrices must have same number of elements for dot product ({len(self)} vs {len(other)})")
        return sum(self[i] * other[i] for i in range(len(self)))

    def cross(self, other):
        """Compute the cross product with another 3-element vector."""
        if not isinstance(other, MatrixBase):
            try:
                other = Matrix(other)
            except Exception:
                raise TypeError(f"Cannot compute cross product with {type(other)}")
        if len(self) != 3 or len(other) != 3:
            raise ValueError("Cross product is only defined for 3-element vectors")
        a1, a2, a3 = self[0], self[1], self[2]
        b1, b2, b3 = other[0], other[1], other[2]
        return Matrix([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])

    def reshape(self, rows, cols):
        """Return a new matrix with specified dimensions containing the same elements."""
        rows = int(rows)
        cols = int(cols)
        if rows * cols != len(self):
            raise ValueError(f"Total elements {len(self)} cannot be reshaped to ({rows}, {cols})")
        flat = [_native_expr(self[i]) for i in range(len(self))]
        return self._new(_NativeMatrix(rows, cols, flat))

    def vec(self):
        """Vectorize the matrix by stacking columns into a column vector."""
        flat = []
        for c in range(self.cols):
            for r in range(self.rows):
                flat.append(_native_expr(self[r, c]))
        return self._new(_NativeMatrix(len(flat), 1, flat))

    def vech(self):
        """Vectorize the lower triangular half of the matrix."""
        flat = []
        for c in range(self.cols):
            for r in range(c, self.rows):
                flat.append(_native_expr(self[r, c]))
        return self._new(_NativeMatrix(len(flat), 1, flat))

    @staticmethod
    def eye(n):
        return eye(n)

    @staticmethod
    def zeros(r, c=None):
        return zeros(r, c)

    @staticmethod
    def ones(r, c=None):
        return ones(r, c)

    @staticmethod
    def diag(*entries):
        return diag(*entries)

    def __repr__(self):
        return f"Matrix({self.tolist()})"

    def __str__(self):
        # Oracle parity: str(Matrix(...)) is the single-line form (finding 6,
        # printer-parity bead qxr); the native multi-line pretty is not str.
        return f"Matrix({self.tolist()})"

    def _repr_latex_(self):
        return self._native._repr_latex_()

    def __eq__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                return False
            return self._native.flat() == other._native.flat()
        if isinstance(other, MatrixBase):
            if hasattr(other, "to_dense"):
                return self == other.to_dense()
        return False


DenseMatrix = Matrix
MutableDenseMatrix = Matrix


class ImmutableDenseMatrix(Matrix):
    """Hashable immutable dense matrix (SymPy 1.14 semantics, finding 13).

    Upstream immutable matrices are hashable content-addressed values; the
    mutable base stays unhashable exactly as in modern SymPy.
    """

    def __hash__(self):
        return hash((self.shape, tuple(self._native.flat())))

    def __setitem__(self, key, value):
        raise TypeError("Cannot set an item on an immutable matrix")

    def col_del(self, j):
        raise TypeError("Cannot delete from an immutable matrix")

    def row_del(self, i):
        raise TypeError("Cannot delete from an immutable matrix")

    def row_swap(self, i, j):
        raise TypeError("Cannot modify an immutable matrix")

    def col_swap(self, i, j):
        raise TypeError("Cannot modify an immutable matrix")

    def row_op(self, i, f):
        raise TypeError("Cannot modify an immutable matrix")

    def col_op(self, j, f):
        raise TypeError("Cannot modify an immutable matrix")

    def copy(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def as_immutable(self):
        return self

    def as_mutable(self):
        return Matrix(self)


ImmutableMatrix = ImmutableDenseMatrix


def eye(n):
    return Matrix(_NativeMatrix.eye(int(n)))


def zeros(r, c=None):
    if c is None:
        c = r
    return Matrix(_NativeMatrix.zeros(int(r), int(c)))


def diag(*entries):
    has_matrix = any(isinstance(e, MatrixBase) for e in entries)
    if not has_matrix:
        flat = []
        for elem in entries:
            if isinstance(elem, (list, tuple)):
                for sub in elem:
                    flat.append(_native_expr(sub))
            else:
                flat.append(_native_expr(elem))
        return Matrix(_NativeMatrix.diag(flat))

    blocks = []
    for elem in entries:
        if isinstance(elem, MatrixBase):
            blocks.append(elem)
        elif isinstance(elem, (list, tuple)):
            for sub in elem:
                if isinstance(sub, MatrixBase):
                    blocks.append(sub)
                else:
                    blocks.append(Matrix([[sub]]))
        else:
            blocks.append(Matrix([[elem]]))

    total_rows = sum(b.rows for b in blocks)
    total_cols = sum(b.cols for b in blocks)
    res = zeros(total_rows, total_cols)
    cur_r = 0
    cur_c = 0
    for b in blocks:
        for r in range(b.rows):
            for c in range(b.cols):
                res[cur_r + r, cur_c + c] = b[r, c]
        cur_r += b.rows
        cur_c += b.cols
    return res


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
    from ..core import simplify, sqrt

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
        w = w.applyfunc(simplify)
        is_zero_vec = True
        for i in range(len(w)):
            if w[i] != 0:
                is_zero_vec = False
                break
        if not is_zero_vec:
            if orthonormal:
                norm_sq = (w.T * w)[0, 0]
                w = w / sqrt(norm_sq)
                w = w.applyfunc(simplify)
            out.append(w)
    return out


def pinv(m):
    """Compute the Moore-Penrose pseudoinverse."""
    if not isinstance(m, MatrixBase):
        raise TypeError("pinv requires Matrix argument")
    return m.pinv()


def jordan_cell(eigenval, n):
    """Create an n x n Jordan cell with eigenval on diagonal and 1 on superdiagonal."""
    n = int(n)
    if n < 0:
        raise ValueError("Jordan cell size must be non-negative")
    res = zeros(n, n)
    for i in range(n):
        res[i, i] = eigenval
        if i + 1 < n:
            res[i, i + 1] = 1
    return res


jordan_block = jordan_cell


def det(m, method=None):
    """Compute determinant of a square matrix."""
    if hasattr(m, "det"):
        return m.det(method=method)
    raise TypeError(f"det expected Matrix argument, got {type(m)}")


def trace(m):
    """Compute trace of a square matrix."""
    if hasattr(m, "trace"):
        return m.trace()
    raise TypeError(f"trace expected Matrix argument, got {type(m)}")


def rank(m):
    """Compute rank of a matrix."""
    if hasattr(m, "rank"):
        return m.rank()
    raise TypeError(f"rank expected Matrix argument, got {type(m)}")


def shape(m):
    """Return (rows, cols) shape of a matrix."""
    if hasattr(m, "shape"):
        return m.shape
    raise TypeError(f"shape expected Matrix argument, got {type(m)}")


def randMatrix(
    r,
    c=None,
    min=0,
    max=99,
    seed=None,
    symmetric=False,
    percent=100,
):
    """Create a random matrix of dimensions r x c with integer elements."""
    import random
    if c is None:
        c = r
    rng = random.Random(seed) if seed is not None else random
    if symmetric:
        if r != c:
            raise ValueError("Symmetric matrices must be square")
        mat = [[0] * c for _ in range(r)]
        for i in range(r):
            for j in range(i, c):
                if percent == 100 or rng.randint(1, 100) <= percent:
                    val = rng.randint(min, max)
                else:
                    val = 0
                mat[i][j] = val
                mat[j][i] = val
        return Matrix(mat)
    else:
        mat = []
        for _ in range(r):
            row = []
            for _ in range(c):
                if percent == 100 or rng.randint(1, 100) <= percent:
                    row.append(rng.randint(min, max))
                else:
                    row.append(0)
            mat.append(row)
        return Matrix(mat)


def cholesky(A, hermitian=True):
    """Return the Cholesky decomposition of matrix A."""
    if not isinstance(A, MatrixBase):
        A = Matrix(A)
    return A.cholesky(hermitian=hermitian)
