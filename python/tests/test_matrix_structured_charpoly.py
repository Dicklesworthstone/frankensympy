"""Characteristic polynomials from exact structured source matrices."""

from math import comb

import pytest

from sympy import (
    ImmutableMatrix, ImmutableSparseMatrix, Matrix, PurePoly, Rational, SparseMatrix, Symbol,
)
from sympy.matrices.exceptions import NonSquareMatrixError


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix])
def test_path_spectrum_beyond_dense_charpoly_limit(cls):
    n = 33
    t = Symbol("t")
    edges = {(i, i + 1): 2 for i in range(n - 1)}
    edges.update({(i + 1, i): Rational(1, 2) for i in range(n - 1)})
    if cls in (Matrix, ImmutableMatrix):
        matrix = cls(n, n, lambda i, j: edges.get((i, j), 0))
    else:
        matrix = cls(n, n, edges)
    result = matrix.charpoly(t)
    expected = [0] * (n + 1)
    for k in range(n // 2 + 1):
        expected[2*k] = (-1)**k * comb(n-k, k)
    assert type(result) is PurePoly
    assert result.gen == t
    assert result.all_coeffs() == expected


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_small_general_block_and_symbolic_generator_collision(cls):
    t = Symbol("t")
    # A weighted directed three-cycle, including unequal rational edge weights.
    cycle = cls([[0, Rational(1, 3), 0], [0, 0, -2], [3, 0, 0]])
    assert cycle.charpoly("t").all_coeffs() == [1, 0, 0, 2]
    symbolic = cls([[t, 0], [1, 2]]).charpoly("t")
    assert symbolic.gen == Symbol("_t")
    assert symbolic.all_coeffs() == [1, -t-2, 2*t]
    assert cls(0, 0, {}).charpoly().all_coeffs() == [1]
    with pytest.raises(NonSquareMatrixError):
        cls(10, 12, {}).charpoly()


def test_sparse_charpoly_uses_the_stored_band(monkeypatch):
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native sparse storage contract")
    def dense_would_lose_storage(*args):
        raise AssertionError("unexpected conversion of sparse tridiagonal matrix")
    monkeypatch.setattr(SparseMatrix, "to_dense", dense_would_lose_storage)
    result = SparseMatrix(40, 40, {(0, 1): 2, (1, 0): 3}).charpoly("z")
    assert result.all_coeffs() == [1, 0, -6] + [0]*38
