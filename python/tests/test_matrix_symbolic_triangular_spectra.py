"""Formal eigenvalue multiplicities of large triangular polynomial matrices."""

import pytest

from sympy import Dummy, ImmutableMatrix, ImmutableSparseMatrix, Matrix, Mul, Pow, Rational, SparseMatrix, Symbol


def _matrix(cls, n, entries):
    if cls in (Matrix, ImmutableMatrix):
        return cls(n, n, lambda i, j: entries.get((i, j), 0))
    return cls(n, n, entries)


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("lower", [False, True])
def test_triangular_polynomial_spectrum_preserves_owners_and_multiplicity(cls, lower):
    x = Symbol("x", positive=True)
    y = Dummy("x")
    first = (x + 1)**2
    n = 37
    cells = {(0, 0): first, (1, 1): x**2 + 2*x + 1, (2, 2): y, (3, 3): x,
             (4, 4): x, (5, 5): Rational(-2, 3)}
    for i in range(n-1):
        coord = (i+1, i) if lower else (i, i+1)
        cells[coord] = (x + y)**2
    result = _matrix(cls, n, cells).eigenvals()
    assert result == {first: 2, y: 1, x: 2, Rational(-2, 3): 1, 0: 31}
    assert any(root is y for root in result)
    assert any(root is x for root in result)


def test_same_named_dummies_and_assumed_symbols_remain_independent_diagonal_roots():
    a, b = Dummy("s"), Dummy("s")
    s, positive = Symbol("s"), Symbol("s", positive=True)
    cells = {(i, i): value for i, value in enumerate([a, b, a, s, positive])}
    result = ImmutableSparseMatrix(45, 45, cells).eigenvals()
    assert result == {a: 2, b: 1, s: 1, positive: 1, 0: 40}


def test_large_nontriangular_symbolic_source_and_hidden_pole_refuse():
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("bounded native symbolic spectrum")
    x = Symbol("x")
    with pytest.raises(NotImplementedError, match="triangular"):
        SparseMatrix(40, 40, {(0, 1): x, (1, 0): 1}).eigenvals()
    hidden_pole = Mul(0, Pow(x, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError), match="power|polynomial"):
        SparseMatrix(40, 40, {(0, 0): x, (0, 1): hidden_pole}).eigenvals()
    assert SparseMatrix([[0, x], [1, 0]]).eigenvals() == {x**Rational(1, 2): 1, -x**Rational(1, 2): 1}


def test_changed_retained_source_is_rejected_before_return(monkeypatch):
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native source ownership")
    from sympy.matrices import _triangular_spectrum

    x = Symbol("x")
    matrix = SparseMatrix(35, 35, {(0, 0): x, (1, 1): x})
    original = _triangular_spectrum._polynomial_key
    def mutate(*args):
        matrix[0, 0] = 99
        return original(*args)
    monkeypatch.setattr(_triangular_spectrum, "_polynomial_key", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.eigenvals()


@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_native_only_storage_restores_the_original_typed_diagonal_owners(cls):
    if not hasattr(SparseMatrix, "_native_storage"):
        pytest.skip("native storage constructor")
    a, b = Symbol("a", positive=True), Dummy("a")
    source = _matrix(cls, 36, {(0, 0): a, (1, 1): b, (2, 1): a + b})
    native_only = cls(source._native_storage)
    result = native_only.eigenvals()
    assert result == {a: 1, b: 1, 0: 34}
    assert any(key is a for key in result)
    assert any(key is b for key in result)
