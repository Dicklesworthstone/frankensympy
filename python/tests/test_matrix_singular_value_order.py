"""Exact rectangular singular-value shapes and rational spectral ordering."""

import pytest

from sympy import (
    I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Rational,
    SparseMatrix, nan, sqrt, zoo,
)


_MATRIX_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
_CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
_native_only = pytest.mark.skipif(
    not _CANDIDATE, reason="native singular-value output limit",
)


@pytest.mark.parametrize("matrix_class", _MATRIX_CLASSES)
def test_wide_spectra_preserve_column_nullity(matrix_class):
    matrix = matrix_class([[1, 0, 0], [0, 2, 0]])
    assert matrix.singular_values() == [2, 1, 0]
    assert matrix.condition_number() == zoo
    assert matrix.T.singular_values() == [2, 1]
    assert matrix.T.condition_number() == 2


@pytest.mark.parametrize("matrix_class", _MATRIX_CLASSES)
def test_one_row_complex_spectrum_includes_every_column_zero(matrix_class):
    matrix = matrix_class([[I, 0, 0]])
    assert matrix.singular_values() == [1, 0, 0]
    assert matrix.condition_number() == zoo
    assert matrix.T.singular_values() == [1]
    assert matrix.T.condition_number() == 1


def test_nondiagonal_wide_gram_retains_radicals_and_zero_padding():
    matrix = Matrix([[1, 1, 0], [0, 1, 0]])
    larger = Rational(3, 2) + sqrt(5) / 2
    smaller = Rational(3, 2) - sqrt(5) / 2
    assert matrix.singular_values() == [sqrt(larger), sqrt(smaller), 0]
    assert matrix.condition_number() == zoo


@pytest.mark.parametrize("matrix_class", _MATRIX_CLASSES)
@pytest.mark.parametrize("rows, cols, expected", [
    (0, 0, []),
    (0, 3, [0, 0, 0]),
    (3, 0, []),
])
def test_zero_area_spectra_have_column_shape_and_zero_condition(
    matrix_class, rows, cols, expected,
):
    matrix = matrix_class(rows, cols, [])
    assert matrix.singular_values() == expected
    assert matrix.condition_number() == 0


@pytest.mark.parametrize("matrix_class", _MATRIX_CLASSES)
def test_zero_and_nonzero_singular_matrices_have_distinct_conditions(matrix_class):
    zero = matrix_class([[0, 0], [0, 0]])
    assert zero.singular_values() == [0, 0]
    assert zero.condition_number() is nan
    nonzero = matrix_class([[1, 0], [0, 0]])
    assert nonzero.condition_number() == zoo


@pytest.mark.parametrize("first, second", [
    (Integer(1), Integer(1) + Rational(1, 10**100)),
    (Integer(10**400), Integer(2 * 10**400)),
])
def test_exact_rational_singular_values_do_not_sort_through_float(
    monkeypatch, first, second,
):
    calls = []

    def forbid_evalf(*args, **kwargs):
        calls.append(True)
        raise AssertionError("rational spectral order must stay exact")

    # The exact-ordering implementation contract is native-specific; the
    # upstream oracle may call evalf while proving unrelated sqrt assumptions.
    if _CANDIDATE:
        monkeypatch.setattr(Rational, "evalf", forbid_evalf)
    matrix = Matrix([[first, 0], [0, second]])
    assert matrix.singular_values() == [second, first]
    assert matrix.condition_number() == second / first
    assert calls == []


def test_rational_gram_order_precedes_irrational_square_roots():
    epsilon = Rational(1, 10**100)
    # Orthogonal columns have the exact squared lengths 2 and 2+2e+e**2.
    # Their irrational square roots coincide at ordinary float precision.
    matrix = Matrix([[1, 0], [1, 0], [0, 1], [0, 1 + epsilon]])
    larger = sqrt(2 + 2 * epsilon + epsilon**2)
    assert matrix.singular_values() == [larger, sqrt(2)]
    assert matrix.condition_number() == larger / sqrt(2)


@_native_only
@pytest.mark.parametrize("matrix_class", (SparseMatrix, ImmutableSparseMatrix))
@pytest.mark.parametrize("rows", (0, 1))
def test_sparse_singular_value_padding_preflights_output_size(matrix_class, rows):
    matrix = matrix_class(rows, 10**9, {})
    with pytest.raises(ValueError, match="singular-value count"):
        matrix.singular_values()
    if rows == 0:
        assert matrix.condition_number() == 0
