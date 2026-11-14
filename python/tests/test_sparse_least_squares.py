"""Checked rational sparse least squares, preserving the pinned result classes."""

import inspect

import pytest

from sympy import (
    Add, Float, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix,
    Mul, Pow, Rational, SparseMatrix, Symbol, sqrt,
)


RHS_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)


def _forbid_dense(*args, **kwargs):
    pytest.fail("sparse least squares attempted a dense source conversion")


def test_default_sparse_method_matches_the_pinned_sparse_contract():
    assert inspect.signature(SparseMatrix.solve_least_squares).parameters["method"].default == "LDL"


@pytest.mark.parametrize("coefficient_type", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
@pytest.mark.parametrize("method", [None, "LDL", "CH"])
def test_coupled_inconsistent_multiple_rhs_and_result_classes(
    coefficient_type, rhs_type, method, monkeypatch,
):
    coefficients = coefficient_type([[1, 0], [0, 1], [1, 1]])
    rhs = rhs_type([[1, 2], [2, -1], [4, 0]])
    original_a = coefficients.todok()
    original_b = [rhs[row, col] for row in range(3) for col in range(2)]
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = (coefficients.solve_least_squares(rhs) if method is None
              else coefficients.solve_least_squares(rhs, method=method))
    expected_type = ImmutableSparseMatrix if coefficient_type is ImmutableSparseMatrix else rhs_type
    assert type(result) is expected_type
    assert result == expected_type([
        [Rational(4, 3), Rational(5, 3)],
        [Rational(7, 3), Rational(-4, 3)],
    ])
    assert coefficients.todok() == original_a
    assert [rhs[row, col] for row in range(3) for col in range(2)] == original_b
    assert coefficients.mutation_generation == rhs.mutation_generation == 0


@pytest.mark.parametrize("coefficient_type", [SparseMatrix, ImmutableSparseMatrix])
def test_large_tall_system_stays_sparse_with_two_inconsistent_rhs(coefficient_type, monkeypatch):
    size = 600
    entries = {(row, row): 1 for row in range(size)}
    entries.update({(size + row, row): 2 for row in range(size)})
    values = {(row, 0): 3 for row in range(size)}
    values.update({(size + row, 0): 4 for row in range(size)})
    values.update({(row, 1): -1 for row in range(size)})
    values.update({(size + row, 1): 3 for row in range(size)})
    coefficients = coefficient_type(2*size, size, entries)
    rhs = SparseMatrix(2*size, 2, values)
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = coefficients.solve_least_squares(rhs)
    assert type(result) is coefficient_type
    assert result.shape == (size, 2)
    expected = {(row, 0): Rational(11, 5) for row in range(size)}
    expected.update({(row, 1): 1 for row in range(size)})
    assert result.todok() == expected
    assert coefficients.nnz == 2*size
    assert rhs.nnz == 4*size


@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
def test_empty_rhs_and_zero_unknowns_preserve_shapes(rhs_type):
    coefficients = SparseMatrix([[1, 0], [0, 1], [1, 1]])
    result = coefficients.solve_least_squares(rhs_type(3, 0, []))
    assert type(result) is rhs_type and result.shape == (2, 0)
    # Unlike an exact solve, nonzero B is admissible when there are no unknowns.
    result = SparseMatrix(3, 0, {}).solve_least_squares(rhs_type([1, -2, 3]))
    assert type(result) is rhs_type and result.shape == (0, 1)
    result = SparseMatrix(0, 0, {}).solve_least_squares(rhs_type(0, 2, []))
    assert type(result) is rhs_type and result.shape == (0, 2)


@pytest.mark.parametrize("rhs_columns", [0, 1])
@pytest.mark.parametrize("coefficients", [
    SparseMatrix([[1, 2], [2, 4], [3, 6]]),
    SparseMatrix([[1, 0, 0], [0, 1, 0]]),
    SparseMatrix(0, 2, {}),
])
def test_rank_deficient_and_underdetermined_sources_refuse(coefficients, rhs_columns):
    with pytest.raises(ValueError, match="singular"):
        coefficients.solve_least_squares(SparseMatrix(coefficients.rows, rhs_columns, {}))


def test_tiny_coefficients_keep_exact_residuals_and_do_not_underflow():
    tiny = Rational(1, 10**400)
    coefficients = SparseMatrix([[tiny], [2*tiny]])
    result = coefficients.solve_least_squares(ImmutableSparseMatrix([1, 1]))
    assert type(result) is ImmutableSparseMatrix
    assert result == ImmutableSparseMatrix([3/(5*tiny)])


@pytest.mark.parametrize("method", ["QR", "LU", "invalid"])
def test_unsupported_methods_are_explicit(method):
    with pytest.raises(NotImplementedError, match="LDL, CH and PINV"):
        SparseMatrix([[1], [2]]).solve_least_squares(Matrix([1, 1]), method=method)


def test_pinv_method_returns_the_exact_unique_solution_when_rank_is_full():
    result = SparseMatrix([[1], [2]]).solve_least_squares(Matrix([1, 1]), method="PINV")
    assert type(result) is Matrix
    assert result == Matrix([Rational(3, 5)])


def test_method_objects_cannot_run_custom_comparisons():
    calls = []

    class Method(str):
        def __eq__(self, other):
            calls.append(other)
            raise AssertionError("method comparison callback was invoked")

    with pytest.raises(TypeError, match="string"):
        SparseMatrix([[1], [2]]).solve_least_squares(Matrix([1, 1]), method=Method("LDL"))
    assert calls == []


@pytest.mark.parametrize("entry", [Symbol("sparse_ls_x"), Float(1.0), I, sqrt(2)])
@pytest.mark.parametrize("source_side", ["coefficients", "rhs"])
def test_nonrational_sources_refuse_without_symbolic_or_dense_fallback(entry, source_side, monkeypatch):
    coefficients = SparseMatrix([[entry if source_side == "coefficients" else 1], [2]])
    rhs = SparseMatrix([entry if source_side == "rhs" else 1, 1])
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="exact rational"):
        coefficients.solve_least_squares(rhs)


def test_retained_typed_expression_cannot_be_laundered_through_rational_native_value():
    coefficients = SparseMatrix([[2], [1]])
    coefficients._surface[(0, 0)] = Add(1, 1, evaluate=False)
    with pytest.raises(NotImplementedError, match="literal exact rational"):
        coefficients.solve_least_squares(Matrix([1, 2]))


def test_held_undefined_rhs_is_checked_even_without_unknowns():
    undefined = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError), match="exact rational"):
        SparseMatrix(2, 0, {}).solve_least_squares(SparseMatrix([undefined, 0]))


@pytest.mark.parametrize("source_side", ["coefficients", "rhs"])
@pytest.mark.parametrize("rhs_type", [Matrix, SparseMatrix])
def test_changed_retained_scalar_on_either_source_is_rejected(source_side, rhs_type):
    from sympy.core import _native

    retained = Integer(19)
    coefficients = SparseMatrix([[retained if source_side == "coefficients" else 19], [1]])
    rhs = rhs_type([retained if source_side == "rhs" else 19, 1])
    original = retained._value
    try:
        retained._value = _native.py_integer(0)
        with pytest.raises(RuntimeError, match="native snapshot"):
            coefficients.solve_least_squares(rhs)
    finally:
        retained._value = original


def test_replaced_native_solver_is_not_invoked(monkeypatch):
    from sympy.core import _native

    coefficients = SparseMatrix([[1], [2]])
    rhs = Matrix([1, 1])
    calls = []

    def changed(*args):
        calls.append(args)
        raise AssertionError("unadmitted native solver was invoked")

    monkeypatch.setattr(_native.SparseMatrix, "solve_least_squares", changed)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        coefficients.solve_least_squares(rhs)
    assert calls == []


@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
def test_result_constructor_changes_refuse_without_invoking_them(rhs_type, monkeypatch):
    coefficients = SparseMatrix([[1], [2]])
    rhs = rhs_type([1, 1])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted result constructor was invoked")

    monkeypatch.setattr(rhs_type, "__init__", changed)
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        coefficients.solve_least_squares(rhs)
    assert calls == []


def test_native_resource_failure_does_not_restart_with_dense_storage(monkeypatch):
    coefficients = SparseMatrix(8193, 1, {(0, 0): 1})
    rhs = SparseMatrix(8193, 1, {(0, 0): 1})
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="bound|limit|8192"):
        coefficients.solve_least_squares(rhs)


def test_rhs_shape_and_custom_matrix_sources_have_explicit_refusals():
    class CustomSparse(SparseMatrix):
        pass

    with pytest.raises(ValueError, match="incompatible number of rows"):
        SparseMatrix([[1], [2]]).solve_least_squares(Matrix([1]))
    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomSparse([[1], [2]]).solve_least_squares(Matrix([1, 1]))
    with pytest.raises(TypeError, match="built-in Matrix"):
        SparseMatrix([[1], [2]]).solve_least_squares(CustomSparse([1, 1]))
    with pytest.raises(TypeError, match="built-in Matrix"):
        SparseMatrix([[1], [2]]).solve_least_squares([1, 1])


def test_native_sparse_bridge_uses_the_checked_lane():
    coefficients = SparseMatrix([[1, 0], [0, 1], [1, 1]])
    rhs = SparseMatrix([1, 2, 4])
    result = SparseMatrix(coefficients._native.solve_least_squares(rhs._native))
    assert result == SparseMatrix([Rational(4, 3), Rational(7, 3)])
