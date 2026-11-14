"""Exact sparse pseudoinverses with pinned result classes and source guards."""

import inspect

import pytest

from sympy import (
    Add, Float, I, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, sqrt,
)


def _forbid_dense(*args, **kwargs):
    pytest.fail("sparse pseudoinverse converted its coefficient matrix to dense storage")


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("entries,expected", [
    ([[1, 2], [2, 4]], [[Rational(1, 25), Rational(2, 25)],
                        [Rational(2, 25), Rational(4, 25)]]),
    ([[1, 0], [0, 1], [1, 1]],
     [[Rational(2, 3), Rational(-1, 3), Rational(1, 3)],
      [Rational(-1, 3), Rational(2, 3), Rational(1, 3)]]),
    ([[1, 0, 1], [0, 1, 1]],
     [[Rational(2, 3), Rational(-1, 3)],
      [Rational(-1, 3), Rational(2, 3)],
      [Rational(1, 3), Rational(1, 3)]]),
    ([[1, 2], [3, 4]], [[-2, 1], [Rational(3, 2), Rational(-1, 2)]]),
])
def test_rank_deficient_tall_wide_and_invertible_results(matrix_type, entries, expected, monkeypatch):
    matrix = matrix_type(entries)
    original = matrix.todok()
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = matrix.pinv()
    expected_type = Matrix if matrix_type is SparseMatrix else ImmutableSparseMatrix
    assert type(result) is expected_type
    assert result == expected_type(expected)
    assert matrix.todok() == original
    assert matrix.mutation_generation == 0
    assert result.mutation_generation == 0
    # Four independently specified Moore-Penrose obligations, including the
    # projector symmetries that a generic generalized inverse does not satisfy.
    assert matrix * result * matrix == matrix
    assert result * matrix * result == result
    assert (matrix * result).T == matrix * result
    assert (result * matrix).T == result * matrix


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 3)])
@pytest.mark.parametrize("method", ["RD", "ED", "invalid", None])
def test_zero_and_empty_matrices_keep_sparse_class_before_method_selection(matrix_type, shape, method):
    result = matrix_type(*shape, {}).pinv(method=method)
    assert type(result) is matrix_type
    assert result.shape == shape[::-1]
    assert result.nnz == 0


def test_default_method_and_effectful_method_objects():
    assert inspect.signature(SparseMatrix.pinv).parameters["method"].default == "RD"
    calls = []

    class Method(str):
        def __eq__(self, other):
            calls.append(other)
            raise AssertionError("unadmitted method callback")

        def __repr__(self):
            raise AssertionError("unadmitted method printer")

    with pytest.raises(ValueError, match="invalid pinv method"):
        SparseMatrix([[1]]).pinv(Method("RD"))
    assert SparseMatrix(2, 3, {}).pinv(Method("RD")).shape == (3, 2)
    assert calls == []
    with pytest.raises(ValueError, match="invalid pinv method 'invalid'"):
        SparseMatrix([[1]]).pinv("invalid")
    with pytest.raises(NotImplementedError, match="RD method"):
        SparseMatrix([[1]]).pinv("ED")


def test_large_rank_deficient_immutable_result_and_proof_products_stay_sparse(monkeypatch):
    size = 600
    entries = {}
    expected = {}
    for row in range(0, size, 2):
        for i, j, value in [(0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 4)]:
            entries[row + i, row + j] = value
            expected[row + i, row + j] = Rational(value, 25)
    matrix = ImmutableSparseMatrix(size, size, entries)
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = matrix.pinv()
    assert type(result) is ImmutableSparseMatrix
    assert result.shape == (size, size)
    assert result.todok() == expected
    assert result.nnz == 2*size


@pytest.mark.parametrize("entry", [Symbol("pinv_x"), Float(1.0), I, sqrt(2)])
def test_nonrational_sources_refuse_without_dense_fallback(entry, monkeypatch):
    matrix = SparseMatrix([[entry, 1]])
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="exact rational"):
        matrix.pinv()


def test_original_retained_nodes_cannot_be_replaced_by_simplified_native_values():
    matrix = SparseMatrix([[2, 1]])
    matrix._surface[(0, 0)] = Add(1, 1, evaluate=False)
    with pytest.raises(NotImplementedError, match="literal exact rational"):
        matrix.pinv()
    undefined = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError), match="exact rational"):
        SparseMatrix([[undefined]]).pinv()


def test_changed_retained_scalar_is_rejected():
    from sympy.core import _native

    retained = Integer(19)
    matrix = SparseMatrix([[retained, 1]])
    original = retained._value
    try:
        retained._value = _native.py_integer(0)
        with pytest.raises(RuntimeError, match="native snapshot"):
            matrix.pinv()
    finally:
        retained._value = original


def test_replaced_native_pseudoinverse_is_not_called(monkeypatch):
    from sympy.core import _native

    matrix = SparseMatrix([[1, 2]])
    calls = []

    def changed(*args):
        calls.append(args)
        raise AssertionError("unadmitted native pseudoinverse")

    monkeypatch.setattr(_native.SparseMatrix, "pinv", changed)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        matrix.pinv()
    assert calls == []


@pytest.mark.parametrize("matrix_type,result_type", [
    (SparseMatrix, Matrix), (ImmutableSparseMatrix, ImmutableSparseMatrix),
])
def test_changed_result_constructors_refuse_without_invoking_them(matrix_type, result_type, monkeypatch):
    matrix = matrix_type([[1, 2]])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted result constructor")

    monkeypatch.setattr(result_type, "__init__", changed)
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        matrix.pinv()
    assert calls == []


def test_source_mutation_during_lifting_prevents_return(monkeypatch):
    from sympy.matrices import sparse

    matrix = SparseMatrix([[1, 2]])
    original_lift = sparse._checked_sparse_lift

    def changed_lift(*args):
        result = original_lift(*args)
        matrix[0, 0] = 3
        return result

    monkeypatch.setattr(sparse, "_checked_sparse_lift", changed_lift)
    with pytest.raises(RuntimeError, match="mutated during sparse linear algebra"):
        matrix.pinv()


def test_dimension_and_verifier_fill_refusals_do_not_retry_dense(monkeypatch):
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="bound|limit|8192"):
        SparseMatrix(8193, 1, {(0, 0): 1}).pinv()
    # Its pseudoinverse has only 513 entries, but AP has 263169 entries:
    # verification must refuse that fill rather than skip a Penrose condition.
    with pytest.raises(ValueError, match="bound|limit|fill|entries"):
        ImmutableSparseMatrix(513, 1, {(row, 0): 1 for row in range(513)}).pinv()


def test_custom_matrix_subclasses_require_supervised_dispatch():
    class CustomSparse(SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomSparse([[1]]).pinv()


def test_native_minimum_norm_bridge_handles_inconsistent_and_underdetermined_sources():
    matrix = SparseMatrix([[1, 2], [2, 4]])
    rhs = SparseMatrix([[1, 2], [0, 1]])
    result = SparseMatrix(matrix._native.minimum_norm_least_squares(rhs._native))
    assert result == SparseMatrix([
        [Rational(1, 25), Rational(4, 25)],
        [Rational(2, 25), Rational(8, 25)],
    ])
    matrix = SparseMatrix([[1, 1]])
    result = SparseMatrix(matrix._native.minimum_norm_least_squares(SparseMatrix([2])._native))
    assert result == SparseMatrix([1, 1])
    assert matrix * result == SparseMatrix([2])
