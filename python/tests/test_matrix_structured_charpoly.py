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


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix])
def test_permuted_block_triangular_source_with_one_way_couplings(cls):
    # A 3-cycle of gain 6 and a 2-cycle of gain -1 share one-way edges
    # with each other and a path of 65 zero-eigenvalue singleton blocks.
    # The nonzero below the original diagonal rules out ordinary triangularity.
    n = 70
    edges = {(67, 3): 2, (3, 14): Rational(1, 2), (14, 67): 6,
             (20, 5): -3, (5, 20): Rational(1, 3), (67, 20): 17}
    singletons = [i for i in range(n) if i not in (67, 3, 14, 20, 5)]
    edges.update({(a, b): -9 for a, b in zip(singletons, singletons[1:])})
    edges[5, singletons[0]] = 13
    if cls in (Matrix, ImmutableMatrix):
        source = cls(n, n, lambda i, j: edges.get((i, j), 0))
    else:
        source = cls(n, n, edges)
    result = source.charpoly()
    # x^65 (x^3 - 6)(x^2 + 1), including the repeated zero eigenvalue.
    assert result.all_coeffs() == [1, 0, 1, -6, 0, -6] + [0]*65


def test_repeated_triangular_diagonal_factors_and_empty_graph():
    n = 96
    entries = {(i, i): -1 for i in range(n)}
    entries.update({(i + 3, i): Rational(1, i + 1) for i in range(n - 3)})
    assert SparseMatrix(n, n, entries).charpoly().all_coeffs() == [comb(n, k) for k in range(n+1)]
    assert SparseMatrix(256, 256, {}).charpoly().all_coeffs() == [1] + [0]*256


@pytest.mark.parametrize("bad", [[[0], [1], [2]], [[0, 1], [1, 2]], [[0, 1]], [[], [0, 1, 2]]])
def test_unsafe_block_partition_is_rejected_before_publishing(monkeypatch, bad):
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native block-structure claim checker")
    from sympy.matrices import _charpoly

    monkeypatch.setattr(_charpoly, "_components", lambda n, entries: bad)
    # 0 and 1 form one strongly connected component. Splitting it leaves
    # an entry below the proposed block diagonal and changes the determinant.
    with pytest.raises(RuntimeError, match="block"):
        SparseMatrix([[0, 1, 0], [2, 0, 0], [0, 0, 0]]).charpoly()


def test_unsupported_irreducible_block_and_oversize_matrix_are_bounded():
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native resource limits")
    n = 34
    edges = {(i, (i+1) % n): 1 for i in range(n)}
    with pytest.raises(NotImplementedError, match="irreducible"):
        SparseMatrix(n, n, edges).charpoly()
    with pytest.raises(ValueError, match="dimension"):
        SparseMatrix(10**7, 10**7, {}).charpoly()
