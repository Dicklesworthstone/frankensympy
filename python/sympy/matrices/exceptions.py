"""Exception classes shared by exact matrix operations."""


class MatrixError(Exception):
    """Base exception for matrix operations."""


class ShapeError(ValueError, MatrixError):
    """Matrix dimensions do not satisfy the requested operation."""


class NonSquareMatrixError(ShapeError):
    """A square matrix is required."""


class NonInvertibleMatrixError(ValueError, MatrixError):
    """The supplied matrix is provably noninvertible in the requested domain."""


__all__ = [
    "MatrixError", "ShapeError", "NonSquareMatrixError", "NonInvertibleMatrixError",
]
