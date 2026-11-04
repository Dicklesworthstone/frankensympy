"""Unique exact rectangular systems through the public matrix solve APIs."""

import pytest

from sympy import Function, ImmutableMatrix, Matrix, Rational, SparseMatrix, Symbol, zeros


@pytest.mark.parametrize("matrix_type", [Matrix, ImmutableMatrix])
@pytest.mark.parametrize("rhs_type", [Matrix, ImmutableMatrix])
@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_tall_multi_rhs_solve_preserves_exact_values_and_matrix_class(matrix_type, rhs_type, method):
    coefficients = matrix_type([[0, 2], [3, 1], [3, 3], [-3, 1]])
    rhs = rhs_type([[6, 10, -4], [9, 2, 10], [15, 12, 6], [-3, 8, -14]])
    original_coefficients, original_rhs = coefficients.copy(), rhs.copy()
    result = getattr(coefficients, method)(rhs)
    assert result == matrix_type([[2, -1, 4], [3, 5, -2]])
    assert type(result) is (rhs_type if method == "LUsolve" else matrix_type)
    assert coefficients*result == rhs
    assert coefficients == original_coefficients and rhs == original_rhs
    assert coefficients.mutation_generation == rhs.mutation_generation == 0


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_tiny_exact_rectangular_pivot_and_residual_are_not_rounded(method):
    tiny = Rational(1, 10**80)
    coefficients = Matrix([[tiny, 0], [0, 1], [tiny, 1]])
    rhs = Matrix([2*tiny, -3, 2*tiny - 3])
    assert getattr(coefficients, method)(rhs) == Matrix([2, -3])
    rhs[2, 0] += tiny
    with pytest.raises(ValueError, match="inconsistent"):
        getattr(coefficients, method)(rhs)


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_rank_deficient_and_inconsistent_systems_do_not_return_a_candidate(method):
    coefficients = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError, match="singular"):
        getattr(coefficients, method)(Matrix([1, 2, 3]))
    with pytest.raises(ValueError, match="inconsistent"):
        getattr(coefficients, method)(Matrix([1, 2, 4]))
    # The first RHS is consistent, but accepting only that column is unsound.
    coefficients = Matrix([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="inconsistent"):
        getattr(coefficients, method)(Matrix([[2, 3], [4, 5], [6, 9]]))


def test_wide_systems_keep_the_pinned_public_refusals():
    for coefficients, rhs in [
        (Matrix([[1, 2, 3]]), Matrix([1])),
        (Matrix(0, 2, []), Matrix(0, 1, [])),
    ]:
        with pytest.raises(ValueError):
            coefficients.solve(rhs)
        with pytest.raises(NotImplementedError, match="Underdetermined"):
            coefficients.LUsolve(rhs)


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_no_variables_has_the_unique_empty_vector_exactly_when_consistent(method):
    coefficients = Matrix(3, 0, [])
    assert getattr(coefficients, method)(zeros(3, 2)) == Matrix(0, 2, [])
    assert getattr(coefficients, method)(Matrix(3, 0, [])) == Matrix(0, 0, [])
    with pytest.raises(ValueError, match="inconsistent"):
        getattr(coefficients, method)(Matrix([0, 0, 1]))


def test_empty_rhs_distinguishes_solve_from_lusolve_without_skipping_rank():
    coefficients = Matrix([[1, 0], [0, 1], [1, 1]])
    rhs = Matrix(3, 0, [])
    with pytest.raises(ValueError):
        coefficients.solve(rhs)
    assert coefficients.LUsolve(rhs) == Matrix(2, 0, [])
    with pytest.raises(ValueError, match="singular"):
        Matrix([[1, 1], [2, 2], [3, 3]]).LUsolve(rhs)


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_sparse_rhs_keeps_the_pinned_public_result_class(method):
    coefficients = ImmutableMatrix([[1, 0], [0, 1], [1, 1]])
    rhs = SparseMatrix([[2, 0], [3, 5], [5, 5]])
    result = getattr(coefficients, method)(rhs)
    assert result == ImmutableMatrix([[2, 0], [3, 5]])
    assert type(result) is (SparseMatrix if method == "LUsolve" else ImmutableMatrix)


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_symbolic_and_opaque_entries_remain_explicitly_unsupported(method):
    x = Symbol("rectangular_parameter")
    coefficients = Matrix([[1, 0], [0, 1], [x, 1]])
    with pytest.raises(ValueError, match="exact rational"):
        getattr(coefficients, method)(Matrix([1, 2, 3]))
    law = type("RectangularLaw", (Function,), {})
    value = law(x)
    rhs = Matrix([1, 2, value])
    with pytest.raises(NotImplementedError, match="opaque Python"):
        getattr(Matrix([[1, 0], [0, 1], [1, 1]]), method)(rhs)
    assert rhs[2, 0] is value and type(rhs[2, 0]) is law
    assert rhs.mutation_generation == 0


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_rectangular_shape_and_resource_failures_remain_explicit(method):
    coefficients = Matrix([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="incompatible number of rows"):
        getattr(coefficients, method)(Matrix([1, 2]))
    with pytest.raises(TypeError, match="Matrix right-hand side"):
        getattr(coefficients, method)([1, 2, 3])
    with pytest.raises(ValueError, match="128"):
        getattr(coefficients, method)(zeros(3, 129))
    with pytest.raises(ValueError, match="aggregate"):
        getattr(zeros(65, 64), method)(zeros(65, 16))
