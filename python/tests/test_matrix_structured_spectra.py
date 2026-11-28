"""Exact eigenvalue multiplicities of rational block-triangular matrices."""

import pytest

from sympy import CRootOf, I, ImmutableMatrix, ImmutableSparseMatrix, Matrix, Rational, SparseMatrix, Symbol
from sympy.matrices.exceptions import NonSquareMatrixError


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix])
def test_defective_and_complex_blocks_keep_algebraic_multiplicity(cls):
    n = 42
    entries = {(i, i): 3 for i in range(4, n)}
    entries.update({(i, i+1): 1 for i in range(4, n-1)})
    entries.update({(0, 2): -1, (2, 0): 1, (1, 3): -2, (3, 1): Rational(1, 2),
                    (0, 1): 8, (3, 4): 1})
    matrix = (cls(n, n, lambda i, j: entries.get((i, j), 0))
              if cls in (Matrix, ImmutableMatrix) else cls(n, n, entries))
    assert matrix.eigenvals() == {-I: 2, I: 2, 3: 38}


def test_real_indexed_roots_compose_with_complex_radicals_across_distinct_blocks():
    x = Symbol("x")
    p = x**5 - 15*x**4 + 66*x**3 - 82*x**2 - 31*x + 54
    entries = {(0, 5): 9, (5, 10): 4, (10, 11): -1, (11, 10): 1}
    for offset in (0, 5):
        for i, d in enumerate((0, 1, 2, 4, 8)):
            entries[offset+i, offset+i] = d
            if i:
                entries[offset+i, offset+i-1] = entries[offset+i-1, offset+i] = 1
    expected = {CRootOf(p, i): 2 for i in range(5)}
    expected.update({I: 1, -I: 1, 0: 31})
    assert ImmutableSparseMatrix(43, 43, entries).eigenvals() == expected


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_empty_symbolic_and_rectangular_sparse_spectra(cls):
    x = Symbol("x")
    assert cls(0, 0, {}).eigenvals() == {}
    assert cls([[x, 1], [0, x]]).eigenvals() == {x: 2}
    with pytest.raises(NonSquareMatrixError):
        cls(5, 4, {}).eigenvals()


def test_incomplete_nonreal_quintic_block_refuses_the_entire_spectrum():
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native exact indexed-root boundary")
    # x**5 - x + 1 has nonreal roots, and a disjoint known eigenvalue 2
    # must never be mistaken for the full spectrum.
    entries = {(i+1, i): 1 for i in range(4)}
    entries.update({(0, 4): -1, (1, 4): 1, (5, 5): 2})
    with pytest.raises(ValueError, match="unsupported|complete"):
        SparseMatrix(6, 6, entries).eigenvals()


@pytest.mark.parametrize("n,scale", [(33, 1), (34, 2)])
def test_sylvester_kac_matrix_has_its_complete_rational_spectrum(n, scale):
    # In the degree n-1 homogeneous-polynomial basis, x*d/dy + y*d/dx
    # is the Sylvester–Kac tridiagonal matrix. Changing variables to x+y
    # and x-y gives the n eigenvalues n-1, n-3, ..., 1-n.
    entries = {(i, i+1): Rational(i+1, scale) for i in range(n-1)}
    entries.update({(i+1, i): Rational(n-1-i, scale) for i in range(n-1)})
    expected = {Rational(k, scale): 1 for k in range(1-n, n, 2)}
    assert SparseMatrix(n, n, entries).eigenvals() == expected


def test_rational_deflation_preserves_the_unsolved_nonreal_polynomial():
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native source-bound rational factor extraction")
    from sympy.matrices._charpoly import _bounded_rational_factors

    matrix = SparseMatrix([[Rational(1, 2), 1, 0, 0], [0, Rational(1, 2), 0, 0],
                           [0, 0, 0, -1], [0, 0, 1, 0]])
    roots, residual = _bounded_rational_factors(
        matrix.charpoly("x"), 4, matrix._native_storage.todok())
    assert roots == [Rational(1, 2), Rational(1, 2)]
    assert residual.all_coeffs() == [1, 0, 1]
