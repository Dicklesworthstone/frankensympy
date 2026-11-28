"""Division-free determinants of polynomial tridiagonal systems."""

from math import comb

import pytest

from sympy import ImmutableMatrix, ImmutableSparseMatrix, Matrix, Poly, Rational, SparseMatrix, Symbol


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix])
def test_polynomial_path_determinant(cls):
    x = Symbol("x")
    n = 12
    # xI minus the adjacency matrix of a path. k disjoint edges give the
    # sign and coefficient of degree n-2k in its characteristic polynomial.
    matrix = cls(n, n, lambda i, j: x if i == j else int(abs(i - j) == 1))
    expected = sum((-1)**k * comb(n-k, k) * x**(n-2*k) for k in range(n//2+1))
    assert matrix.det() == expected


@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_zero_pivots_rational_entries_and_nonlinear_offdiagonals(cls):
    x = Symbol("x")
    n = 11
    rows = [[int(i == j) for j in range(n)] for i in range(n)]
    rows[0][:2] = [0, (1 + x)**2]
    rows[1][:2] = [Rational(2, 3), x + 5]
    rows[-1][-1] = 3*x
    determinant = cls(rows).det()
    assert Poly(determinant, x).all_coeffs() == [-2, -4, -2, 0]
    rows[-1][-1] = 0
    assert cls(rows).det() == 0


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_sparse_polynomial_system_larger_than_dense_storage(cls):
    n = 600
    x = Symbol("x")
    diagonals = {(i, i): 1 for i in range(n)}
    diagonals[0, 0] = x
    diagonals[n-1, n-2] = -1
    diagonals[n-2, n-1] = 2
    assert cls(n, n, diagonals).det() == 3*x


@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_rejects_a_hidden_symbolic_denominator_even_in_a_zero_row_product(cls):
    from sympy import Mul, Pow

    if not hasattr(Matrix([[1]]), "_native_storage"):
        pytest.skip("bounded native polynomial-domain admission")
    x = Symbol("x")
    hidden_pole = Mul(0, Pow(x, -1, evaluate=False), evaluate=False)
    entries = [[0]*9 for _ in range(9)]
    entries[0][0] = x
    entries[8][8] = hidden_pole
    with pytest.raises(ValueError, match="determinant|resource"):
        cls(entries).det()
