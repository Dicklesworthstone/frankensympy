"""Matrices module for FrankenSymPy (WS05, WS10)."""

from .dense import (
    DenseMatrix,
    ImmutableDenseMatrix,
    ImmutableMatrix,
    Matrix,
    MatrixBase,
    MutableDenseMatrix,
    diag,
    eye,
    hadamard_product,
    kronecker_product,
    matrix_multiply_elementwise,
    zeros,
)

__all__ = [
    "MatrixBase",
    "Matrix",
    "DenseMatrix",
    "MutableDenseMatrix",
    "ImmutableMatrix",
    "ImmutableDenseMatrix",
    "eye",
    "zeros",
    "diag",
    "hadamard_product",
    "kronecker_product",
    "matrix_multiply_elementwise",
]
