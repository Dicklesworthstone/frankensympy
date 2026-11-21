"""Matrices module for FrankenSymPy (WS05, WS10)."""

from .dense import (
    DenseMatrix,
    GramSchmidt,
    ImmutableDenseMatrix,
    ImmutableMatrix,
    Matrix,
    MatrixBase,
    MutableDenseMatrix,
    casoratian,
    cholesky,
    det,
    diag,
    eye,
    hadamard_product,
    hstack,
    jacobian,
    jordan_block,
    jordan_cell,
    kronecker_product,
    matrix_multiply_elementwise,
    ones,
    pinv,
    randMatrix,
    rank,
    shape,
    trace,
    vstack,
    wronskian,
    zeros,
)
from .sparse import (
    ImmutableSparseMatrix,
    MutableSparseMatrix,
    SparseMatrix,
)

# Capture the actual built-in owners before user code can alter the interfaces.
from . import _jordan
from . import _definiteness
from . import normalforms
from . import _integer_solve
from . import _inv_mod

__all__ = [
    "DenseMatrix",
    "GramSchmidt",
    "ImmutableDenseMatrix",
    "ImmutableMatrix",
    "ImmutableSparseMatrix",
    "Matrix",
    "MatrixBase",
    "MutableDenseMatrix",
    "MutableSparseMatrix",
    "SparseMatrix",
    "casoratian",
    "cholesky",
    "det",
    "diag",
    "eye",
    "hadamard_product",
    "hstack",
    "jacobian",
    "jordan_block",
    "jordan_cell",
    "kronecker_product",
    "matrix_multiply_elementwise",
    "ones",
    "pinv",
    "randMatrix",
    "rank",
    "shape",
    "trace",
    "vstack",
    "wronskian",
    "zeros",
]
