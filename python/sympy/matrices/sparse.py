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
            return self._new(self._native.solve(sparse_rhs._native))
        return self._new(self.to_dense().solve(sparse_rhs.to_dense()))

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
