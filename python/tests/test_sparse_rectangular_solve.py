"""Checked tall sparse LUsolve and the pinned sparse solve/LUsolve distinction."""

import pytest

from _native_subprocess import isolated_heap_new_probe

from sympy import (
    ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Rational,
    SparseMatrix, Symbol,
)


RHS_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)


def _forbid_dense(*args, **kwargs):
    pytest.fail("sparse LUsolve attempted a dense source conversion")


@pytest.mark.parametrize("coefficient_type", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
@pytest.mark.parametrize("tall", [False, True])
def test_pivoted_multiple_rhs_keep_rhs_class_for_square_and_tall(
    coefficient_type, rhs_type, tall, monkeypatch,
):
    rows = [[0, 2], [3, 1]]
    values = [[1, 6], [Rational(7, 2), -3]]
    if tall:
        rows.append([3, 3])
        values.append([Rational(9, 2), 3])
    coefficients = coefficient_type(rows)
    rhs = rhs_type(values)
    original_a = coefficients.todok()
    original_b = [rhs[row, col] for row in range(rhs.rows) for col in range(rhs.cols)]
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = coefficients.LUsolve(rhs)
    assert type(result) is rhs_type
    assert result == rhs_type([[1, -2], [Rational(1, 2), 3]])
    assert coefficients.todok() == original_a
    assert [rhs[row, col] for row in range(rhs.rows) for col in range(rhs.cols)] == original_b
    assert coefficients.mutation_generation == rhs.mutation_generation == 0


@pytest.mark.parametrize("rhs_type", [SparseMatrix, ImmutableSparseMatrix])
def test_large_overdetermined_multiple_rhs_never_densify(rhs_type, monkeypatch):
    size = 500
    entries = {(row, row): 2 for row in range(size)}
    entries.update({(size + row, row): 1 for row in range(size)})
    values = {(row, 0): 2*(row + 1) for row in range(size)}
    values.update({(size + row, 0): row + 1 for row in range(size)})
    values[(size - 1, 1)] = -14
    values[(2*size - 1, 1)] = -7
    coefficients = ImmutableSparseMatrix(2*size, size, entries)
    rhs = rhs_type(2*size, 2, values)
    expected = {(row, 0): row + 1 for row in range(size)}
    expected[(size - 1, 1)] = -7
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    result = coefficients.LUsolve(rhs)
    assert type(result) is rhs_type and result.shape == (size, 2)
    assert result.todok() == expected
    assert coefficients*result == rhs


@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
def test_empty_rhs_and_zero_unknowns_preserve_shape_and_rank_obligations(rhs_type):
    full_rank = SparseMatrix([[1, 0], [0, 1], [1, 1]])
    result = full_rank.LUsolve(rhs_type(3, 0, []))
    assert type(result) is rhs_type and result.shape == (2, 0)
    deficient = SparseMatrix([[1, 2], [2, 4], [3, 6]])
    with pytest.raises(ValueError, match="singular"):
        deficient.LUsolve(rhs_type(3, 0, []))
    no_unknowns = SparseMatrix(3, 0, {})
    result = no_unknowns.LUsolve(rhs_type([[0, 0], [0, 0], [0, 0]]))
    assert type(result) is rhs_type and result.shape == (0, 2)
    with pytest.raises(ValueError, match="inconsistent"):
        no_unknowns.LUsolve(rhs_type([0, 0, 1]))
    result = SparseMatrix(0, 0, {}).LUsolve(rhs_type(0, 3, []))
    assert type(result) is rhs_type and result.shape == (0, 3)


def test_any_inconsistent_rhs_refuses_the_complete_solution():
    coefficients = SparseMatrix([[1, 0], [0, 1], [1, 1]])
    rhs = SparseMatrix([[2, 7], [3, 8], [5, 16]])
    with pytest.raises(ValueError, match="inconsistent"):
        coefficients.LUsolve(rhs)
    with pytest.raises(ValueError, match="singular"):
        SparseMatrix([[1, 2], [2, 4], [3, 6]]).LUsolve(Matrix([5, 10, 15]))


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_underdetermined_systems_refuse_without_free_parameter_invention(shape):
    with pytest.raises(NotImplementedError, match="Underdetermined"):
        SparseMatrix(*shape, {}).LUsolve(SparseMatrix(shape[0], 1, {}))


def test_rhs_row_mismatch_is_explicit():
    with pytest.raises(ValueError, match="incompatible number of rows"):
        SparseMatrix([[1], [2]]).LUsolve(Matrix([1]))


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (3, 0), (0, 3)])
def test_default_sparse_solve_still_refuses_every_rectangular_shape(shape):
    with pytest.raises(ValueError, match="determined"):
        SparseMatrix(*shape, {}).solve(SparseMatrix(shape[0], 1, {}))


def test_tiny_tall_pivots_are_exact_and_never_binary64():
    tiny = Rational(1, 10**400)
    coefficients = SparseMatrix([[tiny], [2*tiny]])
    result = coefficients.LUsolve(ImmutableSparseMatrix([[1, 3], [2, 6]]))
    assert type(result) is ImmutableSparseMatrix
    assert result == ImmutableSparseMatrix([[1/tiny, 3/tiny]])


def test_symbolic_rectangular_systems_refuse_the_rational_lane():
    x = Symbol("sparse_solve_parameter")
    with pytest.raises(NotImplementedError, match="exact rational"):
        SparseMatrix([[1], [x]]).LUsolve(Matrix([1, x]))


@pytest.mark.parametrize("source_side", ["coefficients", "rhs"])
@pytest.mark.parametrize("rhs_type", [Matrix, SparseMatrix])
def test_changed_retained_scalar_on_either_source_is_rejected(source_side, rhs_type):
    from sympy.core import _native

    retained = Integer(17)
    coefficients = SparseMatrix([[retained if source_side == "coefficients" else 17], [34]])
    rhs = rhs_type([retained if source_side == "rhs" else 17, 34])
    original = retained._value
    try:
        retained._value = _native.py_integer(0)
        with pytest.raises(RuntimeError, match="native snapshot"):
            coefficients.LUsolve(rhs)
    finally:
        retained._value = original


@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
def test_rhs_accessor_shadows_refuse_without_invoking_them(rhs_type, monkeypatch):
    coefficients = SparseMatrix([[1], [2]])
    rhs = rhs_type([3, 6])
    calls = []

    def shadow(self):
        calls.append(self)
        raise AssertionError("changed source accessor was invoked")

    monkeypatch.setattr(rhs_type, "_surface", property(shadow))
    with pytest.raises(NotImplementedError, match="original.*descriptors|original matrix accessors"):
        coefficients.LUsolve(rhs)
    assert calls == []


def test_modified_native_rhs_reader_and_solver_refuse_without_callbacks(monkeypatch):
    from sympy.core import _native

    coefficients = SparseMatrix([[1], [2]])
    rhs = Matrix([3, 6])
    calls = []

    def callback(*args):
        calls.append(args)
        raise AssertionError("changed native interface was invoked")

    with monkeypatch.context() as changed:
        changed.setattr(_native.Matrix, "flat", callback)
        with pytest.raises(NotImplementedError, match="original native interfaces"):
            coefficients.LUsolve(rhs)
    with monkeypatch.context() as changed:
        changed.setattr(_native.SparseMatrix, "solve", callback)
        with pytest.raises(NotImplementedError, match="original native interfaces"):
            coefficients.LUsolve(rhs)
    assert calls == []


@pytest.mark.parametrize("rhs_type", RHS_CLASSES)
@pytest.mark.parametrize("hook", ["__init__", "__new__"])
def test_result_constructors_refuse_without_invocation(rhs_type, hook, monkeypatch, request):
    if isolated_heap_new_probe(request, hook):
        return
    coefficients = SparseMatrix([[1], [2]])
    rhs = rhs_type([3, 6])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted LUsolve result constructor")

    monkeypatch.setattr(rhs_type, hook, changed)
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        coefficients.LUsolve(rhs)
    assert calls == []
    assert coefficients.mutation_generation == rhs.mutation_generation == 0


def test_native_resource_exhaustion_has_no_dense_retry(monkeypatch):
    coefficients = SparseMatrix(8193, 1, {(0, 0): 1})
    rhs = SparseMatrix(8193, 1, {(0, 0): 2})
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="bound|limit|8192"):
        coefficients.LUsolve(rhs)


def test_custom_matrix_classes_and_nonmatrix_rhs_are_explicitly_unsupported():
    class CustomSparse(SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomSparse([[1], [2]]).LUsolve(SparseMatrix([1, 2]))
    with pytest.raises(TypeError, match="built-in Matrix"):
        SparseMatrix([[1], [2]]).LUsolve(CustomSparse([1, 2]))
    with pytest.raises(TypeError, match="built-in Matrix"):
        SparseMatrix([[1], [2]]).LUsolve([1, 2])
