"""Characteristic polynomial dispatch for bounded exact structured matrices."""

from ..core import Dummy, Symbol, _charge_nested_if_active, _native_expr
from ..polys.polytools import PurePoly
from . import sparse
from .exceptions import NonSquareMatrixError


def generator_name(x):
    if x is None:
        return "lambda"
    if type(x) is str:
        return x
    if type(x) in (Symbol, Dummy):
        return str(x)
    raise TypeError("characteristic polynomial generator must be a string, Symbol or Dummy")


def characteristic_polynomial(matrix, x, *, dense):
    """Return det(xI-A) from an exact rational source or bounded symbolic block."""
    rows, cols = matrix.shape
    if rows != cols:
        raise NonSquareMatrixError("Characteristic polynomial requires a square matrix")
    if rows > 64:
        raise ValueError("structured characteristic polynomial supports dimensions up to 64")
    snapshot = sparse._structural_sources(matrix)
    name = generator_name(x)
    _charge_nested_if_active(1 + rows)
    rational = sparse._rational_source_snapshot(
        matrix, dense=dense, context="characteristic polynomial")
    if rational is None:
        if dense or rows > 32:
            raise NotImplementedError(
                "large characteristic polynomial requires literal rational coefficients")
        # Preserve source coefficient identity and generator-name collision rules.
        result = matrix.to_dense().charpoly(x)
    else:
        native, retained = rational
        if dense:
            native = sparse._SOLVE_DENSE_MEMBERS["to_sparse"](native)
        entries = sparse._REDUCTION_NATIVE_MEMBERS["todok"](native)
        gen = Symbol(name)
        if rows and all(abs(row - col) <= 1 for row, col in entries):
            # The independent determinant checker specializes the original
            # shifted entries at degree + 1 points and verifies each sparse LU.
            _charge_nested_if_active(1 + rows * (rows + len(entries)))
            shifted = {ij: -value for ij, value in entries.items()}
            var = _native_expr(gen)
            for i in range(rows):
                shifted[i, i] = var + shifted[i, i] if (i, i) in shifted else var
            polynomial = sparse._NativeSparseMatrix(rows, rows, shifted).tridiagonal_polynomial_det()
            if polynomial is None:
                raise RuntimeError("rational tridiagonal characteristic polynomial was not admitted")
            result = PurePoly(polynomial, gen)
        elif rows <= 32:
            # General small rational matrices use exact shifted-determinant
            # verification, including matrices whose minimal degree is low.
            _charge_nested_if_active(1 + rows**4)
            block = sparse._REDUCTION_NATIVE_MEMBERS["to_dense"](native)
            result = PurePoly.from_list(block.checked_char_poly(), gen)
        else:
            raise NotImplementedError(
                "large characteristic polynomial requires tridiagonal structure")
        sparse._check_rational_source(retained)
    _charge_nested_if_active()
    sparse._check_structural_sources(snapshot)
    if rational is not None:
        sparse._check_rational_source(retained)
    return result
